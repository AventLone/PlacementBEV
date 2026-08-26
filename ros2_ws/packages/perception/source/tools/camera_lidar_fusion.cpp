#include "perception/tools/camera_lidar_fusion.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace
{

// constexpr double kVisibilityDistanceToleranceMeters = 0.05;
constexpr double kVisibilityDistanceToleranceMeters = 0.5;
// Neighborhood (in pixels) treated as belonging to the same ray for occlusion purposes.
constexpr int kVisibilityPixelRadius = 2;

cv::Mat normalizeDistCoeffs(const cv::Mat& dist_coeffs)
{
    if (dist_coeffs.empty())
    {
        return {};
    }

    cv::Mat coeffs_64f;
    dist_coeffs.convertTo(coeffs_64f, CV_64F);
    if (coeffs_64f.rows == 1 || coeffs_64f.cols == 1)
    {
        return coeffs_64f.reshape(1, 1);
    }

    throw std::invalid_argument("dist_coeffs must be a vector-like matrix.");
}

// O(N + W*H) occlusion test: buckets each point's projected depth into its pixel, takes
// the per-pixel minimum over a small neighborhood (emulating the old ray tolerance), and
// marks a point occluded if something closer landed at/near its pixel. Replaces an earlier
// O(N^2) ray-based GPU test that didn't scale past a few thousand points per camera.
std::vector<uint8_t> computePixelVisibilityMask(const std::vector<cv::Point2i>& uvs, const std::vector<double>& depths,
                                                 const cv::Size& image_size)
{
    cv::Mat depth_buffer(image_size, CV_32F, cv::Scalar(std::numeric_limits<float>::max()));
    for (size_t i = 0; i < uvs.size(); ++i)
    {
        if (uvs[i].x < 0)
        {
            continue;
        }
        auto& cell = depth_buffer.at<float>(uvs[i].y, uvs[i].x);
        cell = std::min(cell, static_cast<float>(depths[i]));
    }

    const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(2 * kVisibilityPixelRadius + 1, 2 * kVisibilityPixelRadius + 1));
    cv::Mat min_depth_neighborhood;
    cv::erode(depth_buffer, min_depth_neighborhood, kernel);

    std::vector<uint8_t> mask(uvs.size(), 1U);
    for (size_t i = 0; i < uvs.size(); ++i)
    {
        if (uvs[i].x < 0)
        {
            continue;
        }

        if (const float nearest_depth = min_depth_neighborhood.at<float>(uvs[i].y, uvs[i].x);
            depths[i] > static_cast<double>(nearest_depth) + kVisibilityDistanceToleranceMeters)
        {
            mask[i] = 0U;
        }
    }

    return mask;
}

} // namespace

CameraLidarFusion::CameraLidarFusion(CameraIntrinsics intrinsics, ExtrinsicsLidarToCamera extrinsics)
    : mCameras(1)
{
    mCameras[0].intrinsics = std::move(intrinsics);
    mCameras[0].extrinsics = std::move(extrinsics);

    if (mCameras[0].intrinsics.fx <= 0.0 || mCameras[0].intrinsics.fy <= 0.0)
    {
        throw std::invalid_argument("Camera intrinsics are invalid: fx/fy must be positive.");
    }

    mCameras[0].intrinsics.dist_coeffs = normalizeDistCoeffs(mCameras[0].intrinsics.dist_coeffs);
}

CameraLidarFusion::CameraLidarFusion(std::vector<CameraCalibration> cameras)
    : mCameras(std::move(cameras))
{
    if (mCameras.empty())
    {
        throw std::invalid_argument("At least one camera calibration is required.");
    }

    for (auto& [intrinsics, extrinsics] : mCameras)
    {
        if (intrinsics.fx <= 0.0 || intrinsics.fy <= 0.0)
        {
            throw std::invalid_argument("Camera intrinsics are invalid: fx/fy must be positive.");
        }

        intrinsics.dist_coeffs = normalizeDistCoeffs(intrinsics.dist_coeffs);
    }
}

bool CameraLidarFusion::projectPoint(const pcl::PointXYZ& point_l,
                                     const CameraCalibration& camera,
                                     cv::Point2i& pixel, double& depth,
                                     const bool use_distortion)
{
    const cv::Vec3d p_l(point_l.x, point_l.y, point_l.z);
    const cv::Vec3d p_c = camera.extrinsics.R_cl * p_l + camera.extrinsics.t_cl;
    depth = p_c[2];

    if (depth <= 0.0)
    {
        return false;
    }

    if (use_distortion && !camera.intrinsics.dist_coeffs.empty())
    {
        const std::vector<cv::Point3d> object_points(1, cv::Point3d(point_l.x, point_l.y, point_l.z));
        std::vector<cv::Point2d> image_points;
        cv::Vec3d rvec(0.0, 0.0, 0.0);
        cv::Rodrigues(camera.extrinsics.R_cl, rvec);

        cv::projectPoints(object_points,
                          rvec,
                          camera.extrinsics.t_cl,
                          camera.intrinsics.cameraMatrix(),
                          camera.intrinsics.dist_coeffs,
                          image_points,
                          cv::noArray(),
                          0.0);

        pixel.x = static_cast<int>(std::lround(image_points.front().x));
        pixel.y = static_cast<int>(std::lround(image_points.front().y));
        return true;
    }

    const double u = camera.intrinsics.fx * (p_c[0] / depth) + camera.intrinsics.cx;
    const double v = camera.intrinsics.fy * (p_c[1] / depth) + camera.intrinsics.cy;

    pixel.x = static_cast<int>(std::lround(u));
    pixel.y = static_cast<int>(std::lround(v));
    return true;
}

std::vector<cv::Point2i> CameraLidarFusion::projectToImage(const pcl::PointCloud<pcl::PointXYZ>& lidar_points,
                                                           const cv::Size& image_size,
                                                           const bool use_distortion,
                                                           const bool use_visibility_test) const
{
    if (!hasSingleCamera())
    {
        throw std::invalid_argument("projectToImage(single) requires exactly one camera. Use projectToImages for multi-camera.");
    }

    std::vector<cv::Point2i> pixels(lidar_points.size(), cv::Point2i(-1, -1));
    std::vector<double> depths(lidar_points.size(), 0.0);

    for (size_t point_idx = 0; point_idx < lidar_points.size(); ++point_idx)
    {
        const auto& point = lidar_points[point_idx];
        cv::Point2i uv(-1, -1);
        double depth = 0.0;

        if (!projectPoint(point, mCameras.front(), uv, depth, use_distortion))
        {
            continue;
        }

        if (uv.x < 0 || uv.x >= image_size.width || uv.y < 0 || uv.y >= image_size.height)
        {
            continue;
        }

        pixels[point_idx] = uv;
        depths[point_idx] = depth;
    }

    if (use_visibility_test)
    {
        const std::vector<uint8_t> visibility_mask = computePixelVisibilityMask(pixels, depths, image_size);
        for (size_t point_idx = 0; point_idx < pixels.size(); ++point_idx)
        {
            if (pixels[point_idx].x >= 0 && visibility_mask[point_idx] == 0U)
            {
                pixels[point_idx] = {-1, -1};
            }
        }
    }

    return pixels;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr CameraLidarFusion::colorizePointCloud(const pcl::PointCloud<pcl::PointXYZ>& lidar_points,
                                                                              const cv::Mat& bgr_image,
                                                                              const bool use_distortion,
                                                                              const bool use_visibility_test) const
{
    if (!hasSingleCamera())
    {
        throw std::invalid_argument("colorizePointCloud(single) requires exactly one camera. Use colorizePointCloudMultiCamera for multi-camera.");
    }

    if (bgr_image.empty() || bgr_image.type() != CV_8UC3)
    {
        throw std::invalid_argument("bgr_image must be a non-empty CV_8UC3 image.");
    }

    auto colored_points = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    colored_points->reserve(lidar_points.size());

    const std::vector<cv::Point2i> pixels = projectToImage(lidar_points, bgr_image.size(), use_distortion,
                                                           use_visibility_test);

    for (size_t point_idx = 0; point_idx < lidar_points.size(); ++point_idx)
    {
        const cv::Point2i& uv = pixels[point_idx];
        if (uv.x < 0)
        {
            continue;
        }

        const auto& bgr = bgr_image.at<cv::Vec3b>(uv.y, uv.x);

        pcl::PointXYZRGB out;
        out.x = lidar_points[point_idx].x;
        out.y = lidar_points[point_idx].y;
        out.z = lidar_points[point_idx].z;
        out.r = bgr[2];
        out.g = bgr[1];
        out.b = bgr[0];
        colored_points->push_back(out);
    }

    colored_points->width = static_cast<uint32_t>(colored_points->size());
    colored_points->height = 1;
    colored_points->is_dense = false;

    return colored_points;
}

std::vector<cv::Point2i> CameraLidarFusion::projectToImages(const pcl::PointCloud<pcl::PointXYZ>& lidar_points,
                                                            const std::vector<cv::Size>& image_sizes,
                                                            std::vector<int>& camera_indices,
                                                            const bool use_distortion,
                                                            const bool use_visibility_test) const
{
    if (image_sizes.size() != mCameras.size())
    {
        throw std::invalid_argument("image_sizes size must match number of cameras.");
    }

    const size_t point_count = lidar_points.size();
    std::vector<std::vector<cv::Point2i>> per_camera_uv(mCameras.size(), std::vector<cv::Point2i>(point_count, cv::Point2i(-1, -1)));
    std::vector<std::vector<double>> per_camera_depth(mCameras.size(), std::vector<double>(point_count, 0.0));

    for (size_t cam_idx = 0; cam_idx < mCameras.size(); ++cam_idx)
    {
        for (size_t point_idx = 0; point_idx < point_count; ++point_idx)
        {
            cv::Point2i uv(-1, -1);
            double depth = 0.0;
            if (!projectPoint(lidar_points[point_idx], mCameras[cam_idx], uv, depth, use_distortion))
            {
                continue;
            }

            if (const cv::Size& image_size = image_sizes[cam_idx];
                uv.x < 0 || uv.x >= image_size.width || uv.y < 0 || uv.y >= image_size.height)
            {
                continue;
            }

            per_camera_uv[cam_idx][point_idx] = uv;
            per_camera_depth[cam_idx][point_idx] = depth;
        }
    }

    std::vector<std::vector<uint8_t>> visibility_masks;
    if (use_visibility_test)
    {
        visibility_masks.resize(mCameras.size());
        for (size_t cam_idx = 0; cam_idx < mCameras.size(); ++cam_idx)
        {
            visibility_masks[cam_idx] = computePixelVisibilityMask(per_camera_uv[cam_idx], per_camera_depth[cam_idx],
                                                                    image_sizes[cam_idx]);
        }
    }

    std::vector<cv::Point2i> pixels;
    pixels.reserve(point_count);
    camera_indices.clear();
    camera_indices.reserve(point_count);

    for (size_t point_idx = 0; point_idx < point_count; ++point_idx)
    {
        cv::Point2i best_uv(-1, -1);
        int best_camera = -1;
        double best_depth = std::numeric_limits<double>::max();

        for (size_t cam_idx = 0; cam_idx < mCameras.size(); ++cam_idx)
        {
            const cv::Point2i& uv = per_camera_uv[cam_idx][point_idx];
            if (uv.x < 0)
            {
                continue;
            }

            if (use_visibility_test && visibility_masks[cam_idx][point_idx] == 0U)
            {
                continue;
            }

            if (const double depth = per_camera_depth[cam_idx][point_idx]; depth < best_depth)
            {
                best_depth = depth;
                best_uv = uv;
                best_camera = static_cast<int>(cam_idx);
            }
        }

        pixels.push_back(best_uv);
        camera_indices.push_back(best_camera);
    }

    return pixels;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr CameraLidarFusion::colorizePointCloudMultiCamera(const pcl::PointCloud<pcl::PointXYZ>& lidar_points,
                                                                                          const std::vector<cv::Mat>& bgr_images,
                                                                                          const bool use_distortion,
                                                                                          const bool use_visibility_test) const
{
    if (bgr_images.size() != mCameras.size())
    {
        throw std::invalid_argument("bgr_images size must match number of cameras.");
    }

    std::vector<cv::Size> image_sizes;
    image_sizes.reserve(bgr_images.size());
    for (const auto& image : bgr_images)
    {
        if (image.empty() || image.type() != CV_8UC3)
        {
            throw std::invalid_argument("Each bgr image must be non-empty CV_8UC3.");
        }
        image_sizes.push_back(image.size());
    }

    std::vector<int> camera_indices;
    const std::vector<cv::Point2i> pixels = projectToImages(lidar_points, image_sizes, camera_indices, use_distortion,
                                                            use_visibility_test);

    auto colored_points = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    colored_points->reserve(lidar_points.size());

    for (size_t i = 0; i < lidar_points.size(); ++i)
    {
        const int cam_idx = camera_indices[i];
        if (cam_idx < 0)
        {
            continue;
        }

        const cv::Point2i& uv = pixels[i];
        const cv::Vec3b bgr = bgr_images[static_cast<size_t>(cam_idx)].at<cv::Vec3b>(uv.y, uv.x);

        pcl::PointXYZRGB out;
        out.x = lidar_points[i].x;
        out.y = lidar_points[i].y;
        out.z = lidar_points[i].z;
        out.r = bgr[2];
        out.g = bgr[1];
        out.b = bgr[0];
        colored_points->push_back(out);
    }

    colored_points->width = static_cast<uint32_t>(colored_points->size());
    colored_points->height = 1;
    colored_points->is_dense = false;

    return colored_points;
}

bool CameraLidarFusion::hasSingleCamera() const
{
    return mCameras.size() == 1;
}
