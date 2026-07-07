#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp> // ROS 2 header
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include "perception/tools/2d/FeatureMatchTracking.h"

class Preprocess final : public rclcpp::Node
{
    /* K = [fx 0 cx; 0 fy cy; 0 0 1] */
    static constexpr float cx = 480.0f;
    static constexpr float cy = 300.0f;
    static constexpr float fx = 367.99997615814306f;
    static constexpr float fy = fx;
    static constexpr float fx_inv = 1.0 / fx;
    static constexpr float fy_inv = 1.0 / fy;

    const std::string global_frame_id = "LOLA";

    // struct ImgSet
    // {
    //     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    //     Eigen::Isometry3f T_body2fork;
    //     cv::Mat left_img, right_img;
    // };

    struct ImgSet
    {
        cv::Mat left_rgb, left_semantic;
        cv::Mat right_rgb, right_semantic;

        cv::Mat left_fork_rgb, left_fork_semantic;
        cv::Mat right_fork_rgb, right_fork_semantic;
    };

public:
    explicit Preprocess(const std::string& name, const rclcpp::NodeOptions& options) : rclcpp::Node(name, options),
                                                                                       mT_fork2camera(Eigen::Isometry3f::Identity())
    {
        initSubscriptions();
        initPublishers();

        mT_fork2camera.rotate(Eigen::AngleAxisf(M_PIf, Eigen::Vector3f::UnitZ()));
        mT_fork2camera.prerotate(Eigen::AngleAxisf(-M_PIf / 7.0f, Eigen::Vector3f::UnitY()));
        mT_fork2camera.pretranslate(Eigen::Vector3f(-0.35f, -0.03f, 0.56f));

        // Force the node to use simulation time
        this->set_parameter(rclcpp::Parameter("use_sim_time", true));
        RCLCPP_INFO(this->get_logger(), "Current Sim Time: %f", this->now().seconds());
        mMainWorker = std::thread(&Preprocess::workerLoop, this);
        RCLCPP_INFO(get_logger(), "The node has been activated.");
    }

    ~Preprocess() override
    {
        //
        {
            std::unique_lock lock(mImgBufferMutex);
            mIsShutdown = true;
        }
        mCameraFrameReceived.notify_one();
        if (mMainWorker.joinable())
        {
            mMainWorker.join();
        }
        RCLCPP_INFO(get_logger(), "The node has been shutdown.");
    }

private:
    /* Received Data Buffer */
    bool mIsShutdown{false};
    std::mutex mImgBufferMutex;
    std::condition_variable mCameraFrameReceived;
    std::thread mMainWorker;
    std::queue<ImgSet> mImgsBuffer;

    Eigen::Isometry3f mT_fork2camera;

    /*** Synchronized Subsribers ***/
    using ImgMsg = sensor_msgs::msg::Image;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<ImgMsg, ImgMsg, ImgMsg, ImgMsg, ImgMsg, ImgMsg, ImgMsg, ImgMsg>;
    message_filters::Subscriber<ImgMsg> mLeftRgbSub, mLeftSemanticSub, mRightRgbSub, mRightSemanticSub,
            mLeftForkRgbSub, mLeftForkSemanticSub, mRightForkRgbSub, mRightForkSemanticSub;
    std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> mSynchronizer;

    /* Publishers */
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mBevMapPub;

    void initSubscriptions();

    void initPublishers();

    void imgsHandler(const ImgMsg::ConstSharedPtr& left_rgb_msg, const ImgMsg::ConstSharedPtr& left_semantic_msg,
                     const ImgMsg::ConstSharedPtr& right_rgb_msg, const ImgMsg::ConstSharedPtr& right_semantic_msg,
                     const ImgMsg::ConstSharedPtr& left_fork_rgb_msg, const ImgMsg::ConstSharedPtr& left_fork_semantic_msg,
                     const ImgMsg::ConstSharedPtr& right_fork_rgb_msg, const ImgMsg::ConstSharedPtr& right_fork_semantic_msg);

    void pushInBuffer(ImgSet&& img_set)
    {
        std::lock_guard lock(mImgBufferMutex);
        while (!mImgsBuffer.empty())
        {
            mImgsBuffer.pop();
        }
        mImgsBuffer.push(std::move(img_set));
    }

    void workerLoop();
};
