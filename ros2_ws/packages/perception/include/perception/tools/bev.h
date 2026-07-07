#pragma once
#include <opencv2/opencv.hpp>

struct BevConfig
{
    double x_min = -5.0;
    double x_max = 5.0;
    double y_min = 0.0;
    double y_max = 10.0;
    double resolution = 0.05; // meter / pixel
};

struct BevResult
{
    cv::Mat bev_image; // CV_8UC3
    cv::Mat bev_ground_mask; // CV_8UC1
    cv::Mat valid_mask; // CV_8UC1
    cv::Mat map_x; // CV_32FC1
    cv::Mat map_y; // CV_32FC1
};

inline cv::Matx33d matToMatx33d(const cv::Mat& mat)
{
    CV_Assert(mat.rows == 3 && mat.cols == 3 && mat.type() == CV_64F);

    cv::Matx33d out;
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            out(r, c) = mat.at<double>(r, c);
        }
    }
    return out;
}

inline cv::Vec3d matToVec3d(const cv::Mat& mat)
{
    CV_Assert(mat.total() == 3 && mat.type() == CV_64F);

    cv::Mat t = mat.reshape(1, 3);
    return {t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0)};
}

inline BevResult ipmInverseMapping(const cv::Mat& image, const cv::Mat& ground_mask, const cv::Matx33d& K,
                                   const cv::Matx33d& Rcw, const cv::Vec3d& tcw, const BevConfig& cfg)
{
    CV_Assert(image.type() == CV_8UC3);
    CV_Assert(ground_mask.type() == CV_8UC1);
    CV_Assert(image.rows == ground_mask.rows);
    CV_Assert(image.cols == ground_mask.cols);

    const int bev_w = static_cast<int>((cfg.x_max - cfg.x_min) / cfg.resolution);
    const int bev_h = static_cast<int>((cfg.y_max - cfg.y_min) / cfg.resolution);

    cv::Mat map_x(bev_h, bev_w, CV_32FC1, cv::Scalar(-1.0f));
    cv::Mat map_y(bev_h, bev_w, CV_32FC1, cv::Scalar(-1.0f));
    cv::Mat geometry_valid(bev_h, bev_w, CV_8UC1, cv::Scalar(0));

    for (int row = 0; row < bev_h; ++row)
    {
        for (int col = 0; col < bev_w; ++col)
        {
            double x = cfg.x_min + (col + 0.5) * cfg.resolution;
            double y = cfg.y_max - (row + 0.5) * cfg.resolution;

            cv::Vec3d p_w(x, y, 0.0);
            cv::Vec3d p_c = Rcw * p_w + tcw; // world -> camera

            if (p_c[2] <= 1e-6)
            {
                continue;
            }

            cv::Vec3d p = K * p_c; // camera -> image

            double u = p[0] / p[2];
            double v = p[1] / p[2];

            if (u < 0.0 || u >= image.cols || v < 0.0 || v >= image.rows)
            {
                continue;
            }

            map_x.at<float>(row, col) = static_cast<float>(u);
            map_y.at<float>(row, col) = static_cast<float>(v);
            geometry_valid.at<uchar>(row, col) = 255;
        }
    }

    cv::Mat bev_image;
    cv::Mat bev_ground_mask;

    cv::remap(image, bev_image, map_x, map_y, cv::INTER_LINEAR);
    cv::remap(ground_mask, bev_ground_mask, map_x, map_y, cv::INTER_NEAREST);

    // 只保留：几何投影有效 && 原图中属于 ground mask 的区域
    cv::Mat valid_mask;
    cv::bitwise_and(geometry_valid, bev_ground_mask, valid_mask);

    cv::Mat invalid_mask;
    cv::bitwise_not(valid_mask, invalid_mask);

    bev_image.setTo(cv::Scalar(0, 0, 0), invalid_mask);
    bev_ground_mask.setTo(0, invalid_mask);

    return {bev_image, bev_ground_mask, valid_mask, map_x, map_y};
}

inline cv::Mat getBev(const cv::Mat& img, const cv::Matx33d& K,
                      const cv::Matx33d& Rcw, const cv::Vec3d& tcw, const BevConfig& cfg)
{
    /* 1. 计算BEV图像尺寸 */
    const int bev_w = static_cast<int>((cfg.x_max - cfg.x_min) / cfg.resolution);
    const int bev_h = static_cast<int>((cfg.y_max - cfg.y_min) / cfg.resolution);

    /* 2. 计算单应性矩阵 H (BEV物理坐标 -> 图像像素坐标) */
    const cv::Matx33d H_ground_to_img(K * cv::Matx33d(
                                          Rcw(0, 0), Rcw(0, 1), tcw[0],
                                          Rcw(1, 0), Rcw(1, 1), tcw[1],
                                          Rcw(2, 0), Rcw(2, 1), tcw[2]
                                      ));
    const cv::Matx33d H_bev_to_ground(0.0, -cfg.resolution, cfg.x_max - 0.5 * cfg.resolution,
                                      cfg.resolution, 0.0, cfg.y_min + 0.5 * cfg.resolution,
                                      0.0, 0.0, 1.0);
    const cv::Matx33d H_bev_to_img = H_ground_to_img * H_bev_to_ground;

    /* 3. 执行反向映射 */
    cv::Mat bev;
    cv::warpPerspective(img, bev, H_bev_to_img, cv::Size(bev_w, bev_h), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);
    return bev;
}

struct CameraModel
{
    cv::Matx33d K;
    cv::Matx33d Rcw; // world(or forklift) -> camera
    cv::Vec3d tcw;
    cv::Mat image;
};

inline std::pair<cv::Mat, cv::Mat> bevFusionV2(const std::vector<CameraModel>& cams, const BevConfig& cfg)
{
    const int bev_w = std::ceil((cfg.x_max - cfg.x_min) / cfg.resolution);
    const int bev_h = std::ceil((cfg.y_max - cfg.y_min) / cfg.resolution);

    cv::Mat bev(bev_h, bev_w, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat bev_binary = cv::Mat::zeros(bev_h, bev_w, CV_8UC1);

    for (int r = 0; r < bev_h; r++)
    {
        for (int c = 0; c < bev_w; c++)
        {
            const double x = cfg.x_min + (c + 0.5) * cfg.resolution;
            const double y = cfg.y_max - (r + 0.5) * cfg.resolution;
            cv::Vec3d p_w(x, y, 0.0);

            for (const auto& [K, Rcw, tcw, image] : cams)
            {
                cv::Vec3d p_c = Rcw * p_w + tcw;

                if (p_c[2] <= 1e-3)
                {
                    continue;
                }

                cv::Vec3d p = K * p_c;
                const int u = static_cast<int>(std::round(p[0] / p[2]));
                const int v = static_cast<int>(std::round(p[1] / p[2]));

                if (u < 0 || u >= image.cols || v < 0 || v >= image.rows)
                {
                    continue;
                }

                // 优化内存访问，避免重复调用 ptr
                // if (auto& bev_pixel = bev.ptr<cv::Vec3b>(r)[c]; bev_pixel == cv::Vec3b(0, 0, 0))
                // {
                //     if (const auto& img_pixel = image.ptr<cv::Vec3b>(v)[u]; img_pixel != cv::Vec3b(0, 0, 0))
                //     {
                //         bev_pixel = img_pixel;
                //         bev_binary.ptr<uchar>(r)[c] = 255;
                //     }
                // }

                if (const auto& img_pixel = image.ptr<cv::Vec3b>(v)[u]; img_pixel != cv::Vec3b(0, 0, 0))
                {
                    bev.ptr<cv::Vec3b>(r)[c] = img_pixel;
                    bev_binary.ptr<uchar>(r)[c] = 255;
                }
            }
        }
    }

    return {bev, bev_binary};
}

//生成可放置区域（Minkowski erosion）
inline cv::Mat computeFeasibleRegion(const cv::Mat& free_space, const cv::Size box_size)
{
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, box_size);
    cv::Mat feasible;
    cv::erode(free_space, feasible, kernel); // freeSpace: 1 = free, 0 = obstacle
    return feasible;
}

//距离场（贴边评分用）
inline cv::Mat computeDistanceField(const cv::Mat& free_space)
{
    const cv::Mat inv = 255 - free_space; // obstacle = 1
    cv::Mat dist;
    cv::distanceTransform(inv, dist, cv::DIST_L2, 5);
    return dist;
}

struct Pose2D
{
    int x;
    int y;
    double score;
};

inline Pose2D findBestPlacement(const cv::Mat& feasible, const cv::Mat& dist_field, const float lambda_edge = 1.0f)
{
    Pose2D best{0, 0, -1e9};

    for (int y = 0; y < feasible.rows; y++)
    {
        for (int x = 0; x < feasible.cols; x++)
        {
            if (feasible.ptr<uchar>(y)[x] == 0)
                continue;

            // box center or left-bottom depends on your convention
            // int cx = x;
            // int cy = y;

            // score = "closer to boundary is better"
            const float edge_score = dist_field.at<float>(y, x);

            // optional: encourage packing near walls
            const int boundary_bias = std::min({x, y, feasible.cols - x, feasible.rows - y});

            if (const float score = -lambda_edge * edge_score + static_cast<float>(boundary_bias) * 0.5f; score > best.score)
            {
                best = {x, y, score};
            }
        }
    }

    return best;
}
