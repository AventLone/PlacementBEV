#pragma once
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

struct CameraIntrinsics
{
	double fx{0.0};
	double fy{0.0};
	double cx{0.0};
	double cy{0.0};
	cv::Mat dist_coeffs;

	[[nodiscard]] cv::Matx33d cameraMatrix() const
	{
		return {fx, 0.0, cx,
			    0.0, fy, cy,
			    0.0, 0.0, 1.0};
	}
};

// p_c = R_cl * p_l + t_cl
struct ExtrinsicsLidarToCamera
{
	cv::Matx33d R_cl = cv::Matx33d::eye();
	cv::Vec3d t_cl{0.0, 0.0, 0.0};
};

struct CameraCalibration
{
	CameraIntrinsics intrinsics;
	ExtrinsicsLidarToCamera extrinsics;
};

class CameraLidarFusion
{
public:
	CameraLidarFusion(CameraIntrinsics intrinsics, ExtrinsicsLidarToCamera extrinsics);
	explicit CameraLidarFusion(std::vector<CameraCalibration> cameras);

	[[nodiscard]] pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorizePointCloud(const pcl::PointCloud<pcl::PointXYZ>& lidar_points,
		                                                                    const cv::Mat& bgr_image,
		                                                                    bool use_distortion = true,
		                                                                    bool use_visibility_test = true) const;

	[[nodiscard]] pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorizePointCloudMultiCamera(const pcl::PointCloud<pcl::PointXYZ>& lidar_points,
		                                                                               const std::vector<cv::Mat>& bgr_images,
		                                                                               bool use_distortion = true,
		                                                                               bool use_visibility_test = true) const;

private:
    [[nodiscard]] std::vector<cv::Point2i> projectToImage(const pcl::PointCloud<pcl::PointXYZ>& lidar_points,
                                                          const cv::Size& image_size,
														  bool use_distortion,
														  bool use_visibility_test) const;

    [[nodiscard]] std::vector<cv::Point2i> projectToImages(const pcl::PointCloud<pcl::PointXYZ>& lidar_points,
                                                           const std::vector<cv::Size>& image_sizes,
                                                           std::vector<int>& camera_indices,
														   bool use_distortion,
														   bool use_visibility_test) const;

	static bool projectPoint(const pcl::PointXYZ& point_l, const CameraCalibration& camera,
		                     cv::Point2i& pixel, double& depth, bool use_distortion);

	[[nodiscard]] bool hasSingleCamera() const;

	std::vector<CameraCalibration> mCameras;
};
