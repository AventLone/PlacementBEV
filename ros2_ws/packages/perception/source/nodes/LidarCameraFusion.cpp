#include "perception/nodes/LidarCameraFusion.h"

#include <cv_bridge/cv_bridge.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>

namespace
{

ExtrinsicsLidarToCamera toExtrinsics(const geometry_msgs::msg::TransformStamped& transform)
{
	const Eigen::Isometry3d eigen_transform = tf2::transformToEigen(transform);
	const Eigen::Matrix3d rotation = eigen_transform.rotation();
	const Eigen::Vector3d translation = eigen_transform.translation();

	ExtrinsicsLidarToCamera extrinsics;
	for (int row = 0; row < 3; ++row)
	{
		for (int column = 0; column < 3; ++column)
		{
			extrinsics.R_cl(row, column) = rotation(row, column);
		}
	}
	extrinsics.t_cl = cv::Vec3d(translation.x(), translation.y(), translation.z());
	return extrinsics;
}

} // namespace

LidarCameraFusionNode::LidarCameraFusionNode(const rclcpp::NodeOptions& options)
	: rclcpp::Node("lidar_camera_fusion", options),
	  mTfBuffer(this->get_clock()),
	  mTfListener(mTfBuffer),
	  mLidarFrame(declare_parameter("lidar_frame", "JT128")),
	  mLeftCameraFrame(declare_parameter("left_camera_frame", "camera_left")),
	  mRightCameraFrame(declare_parameter("right_camera_frame", "camera_right")),
	  mUseDistortion(declare_parameter("use_distortion", false))
{
	mIntrinsics.fx = declare_parameter("camera_fx", 367.99997615814306);
	mIntrinsics.fy = declare_parameter("camera_fy", mIntrinsics.fx);
	mIntrinsics.cx = declare_parameter("camera_cx", 480.0);
	mIntrinsics.cy = declare_parameter("camera_cy", 300.0);

	const std::string lidar_topic = declare_parameter("lidar_topic", "/sim_scan");
	const std::string left_image_topic = declare_parameter("left_image_topic", "/cameras/body_left");
	const std::string right_image_topic = declare_parameter("right_image_topic", "/cameras/body_right");
	const std::string fused_topic = declare_parameter("fused_cloud_topic", "/lidar_camera/fused_points");

	mFusedCloudPub = create_publisher<CloudMsg>(fused_topic, rclcpp::SensorDataQoS());

	mLidarSub.subscribe(this, lidar_topic, rmw_qos_profile_sensor_data);
	mLeftImageSub.subscribe(this, left_image_topic, rmw_qos_profile_sensor_data);
	mRightImageSub.subscribe(this, right_image_topic, rmw_qos_profile_sensor_data);

	mSynchronizer = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10),
	                                                                            mLidarSub, mLeftImageSub, mRightImageSub);
	mSynchronizer->registerCallback(std::bind(&LidarCameraFusionNode::imagesHandler, this,
				                    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
	mWorker = std::thread(&LidarCameraFusionNode::workerLoop, this);

    RCLCPP_INFO(get_logger(), "This node has been activated.");
}

LidarCameraFusionNode::~LidarCameraFusionNode()
{
	{
		std::lock_guard lock(mQueueMutex);
		mShutdown = true;
	}
	mQueueCondition.notify_one();
	if (mWorker.joinable())
	{
		mWorker.join();
	}
    RCLCPP_INFO(get_logger(), "This node has been shutdown.");
}

ExtrinsicsLidarToCamera LidarCameraFusionNode::lookupExtrinsics(const std::string& camera_frame, const rclcpp::Time& stamp) const
{
	const auto transform = mTfBuffer.lookupTransform(camera_frame, mLidarFrame, stamp, tf2::durationFromSec(0.05));
	return toExtrinsics(transform);
}

void LidarCameraFusionNode::imagesHandler(const CloudMsg::ConstSharedPtr& lidar_msg,
										  const ImageMsg::ConstSharedPtr& left_image_msg,
										  const ImageMsg::ConstSharedPtr& right_image_msg)
{
	{
		std::lock_guard lock(mQueueMutex);
		mFrameQueue.clear();
		mFrameQueue.push_back({lidar_msg, left_image_msg, right_image_msg});
	}
	mQueueCondition.notify_one();
}

void LidarCameraFusionNode::workerLoop()
{
	while (true)
	{
		FrameSet frame;
		{
			std::unique_lock lock(mQueueMutex);
			mQueueCondition.wait(lock, [this] { return mShutdown || !mFrameQueue.empty(); });

			if (mShutdown && mFrameQueue.empty())
			{
				return;
			}

			frame = std::move(mFrameQueue.front());
			mFrameQueue.pop_front();
		}

		processFrame(frame);
	}
}

void LidarCameraFusionNode::processFrame(const FrameSet& frame)
{
	const auto& lidar_msg = frame.lidar;
	const auto& left_image_msg = frame.left_image;
	const auto& right_image_msg = frame.right_image;

	if (lidar_msg->header.frame_id != mLidarFrame)
	{
		RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
							 "Expected LiDAR frame '%s', received '%s'. TF lookup will use the configured frame.",
							 mLidarFrame.c_str(), lidar_msg->header.frame_id.c_str());
	}

	try
	{
		pcl::PointCloud<pcl::PointXYZ> lidar_points;
		pcl::fromROSMsg(*lidar_msg, lidar_points);

		const cv::Mat left_bgr = cv_bridge::toCvCopy(left_image_msg, "bgr8")->image;
		const cv::Mat right_bgr = cv_bridge::toCvCopy(right_image_msg, "bgr8")->image;

		const rclcpp::Time stamp(lidar_msg->header.stamp);
		// const std::vector<CameraCalibration> cameras{{mIntrinsics, lookupExtrinsics(mLeftCameraFrame, stamp)},
  //                                                    {mIntrinsics, lookupExtrinsics(mRightCameraFrame, stamp)}};

		// const CameraLidarFusion fusion(cameras);
		const CameraLidarFusion fusion_left(mIntrinsics, lookupExtrinsics(mLeftCameraFrame, stamp));
		const CameraLidarFusion fusion_right(mIntrinsics, lookupExtrinsics(mRightCameraFrame, stamp));

	    pcl::PointCloud<pcl::PointXYZ> front_left_cloud, front_right_cloud;
	    front_left_cloud.reserve(lidar_points.size() / 2);
	    front_right_cloud.reserve(lidar_points.size() / 2);
	    for (const auto& point : lidar_points)
	    {
	        if (point.x > 0.0f)
	        {
	            // front_left_cloud.emplace_back(point);
	            continue;
	        }

	        if (point.y > 0.1f && point.y < 8.0f)
	        {
	            front_right_cloud.emplace_back(point);
	        }
	        else if (point.y < -0.1f && point.y > -8.0f)
	        {
	            front_left_cloud.emplace_back(point);
	        }
	    }
	    // RCLCPP_INFO(get_logger(), "Size of original cloud is %lu, while filtered is %lu.", lidar_points.size(), filtered_points.size());
	    // const auto fused_points = fusion.colorizePointCloudMultiCamera(filtered_points, {left_bgr, right_bgr}, mUseDistortion);
	    const auto fused_points_left = fusion_left.colorizePointCloud(front_left_cloud, left_bgr, mUseDistortion);
		const auto fused_points_right = fusion_right.colorizePointCloud(front_right_cloud, right_bgr, mUseDistortion);
	    const auto fused_points = *fused_points_left + *fused_points_right;

		CloudMsg output;
		pcl::toROSMsg(fused_points, output);
		output.header = lidar_msg->header;
		output.header.frame_id = mLidarFrame;
		mFusedCloudPub->publish(output);
	}
	catch (const tf2::TransformException& exception)
	{
		RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
							 "Could not transform %s to a camera frame: %s",
							 mLidarFrame.c_str(), exception.what());
	}
	catch (const cv_bridge::Exception& exception)
	{
		RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
							 "Could not convert camera images: %s", exception.what());
	}
	catch (const std::exception& exception)
	{
		RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
							 "LiDAR-camera fusion failed: %s", exception.what());
	}
}

