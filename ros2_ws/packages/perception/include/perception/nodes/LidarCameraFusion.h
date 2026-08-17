#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "perception/tools/camera_lidar_fusion.h"

class LidarCameraFusionNode final : public rclcpp::Node
{
public:
	explicit LidarCameraFusionNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
	~LidarCameraFusionNode() override;

private:
	using ImageMsg = sensor_msgs::msg::Image;
	using CloudMsg = sensor_msgs::msg::PointCloud2;
	using SyncPolicy = message_filters::sync_policies::ApproximateTime<CloudMsg, ImageMsg, ImageMsg>;

	struct FrameSet
	{
		CloudMsg::ConstSharedPtr lidar;
		ImageMsg::ConstSharedPtr left_image;
		ImageMsg::ConstSharedPtr right_image;
	};

	message_filters::Subscriber<CloudMsg> mLidarSub;
	message_filters::Subscriber<ImageMsg> mLeftImageSub;
	message_filters::Subscriber<ImageMsg> mRightImageSub;
	std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> mSynchronizer;

	rclcpp::Publisher<CloudMsg>::SharedPtr mFusedCloudPub;
	tf2_ros::Buffer mTfBuffer;
	tf2_ros::TransformListener mTfListener;
	std::mutex mQueueMutex;
	std::condition_variable mQueueCondition;
	std::deque<FrameSet> mFrameQueue;
	std::thread mWorker;
	bool mShutdown{false};

	CameraIntrinsics mIntrinsics;
	std::string mLidarFrame;
	std::string mLeftCameraFrame;
	std::string mRightCameraFrame;
	bool mUseDistortion;

	void imagesHandler(const CloudMsg::ConstSharedPtr& lidar_msg,
					   const ImageMsg::ConstSharedPtr& left_image_msg,
					   const ImageMsg::ConstSharedPtr& right_image_msg);

	void workerLoop();

	void processFrame(const FrameSet& frame);

	[[nodiscard]] ExtrinsicsLidarToCamera lookupExtrinsics(const std::string& camera_frame,
															const rclcpp::Time& stamp) const;
};
