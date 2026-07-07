#include "perception/nodes/Preprocess.h"
#include <cv_bridge/cv_bridge.hpp>
#include <Eigen/Geometry>
#include <algorithm>
#include <map>
#include <ranges>
#include <string>
#include "perception/tools/bev.h"

enum
{
    PALLET = 0,
    STORAGE_CAGE = 1,
    GOODS = 2
};

void Preprocess::initSubscriptions()
{
    const std::string params_prefix = "TopicName.Sensor.Camera";
    this->declare_parameters<std::string>(params_prefix, {
                                              {"Body.Left", "/cameras/left/rgb"},
                                              {"Body.Right", "/cameras/right/rgb"},
                                              {"Body.LeftSemantic", "/cameras/left/semantic"},
                                              {"Body.RightSemantic", "/cameras/right/semantic"},
                                              {"Fork.Left", "/cameras/left_fork/rgb"},
                                              {"Fork.LeftSemantic", "/cameras/left_fork/semantic"},
                                              {"Fork.Right", "/cameras/right_fork/rgb"},
                                              {"Fork.RightSemantic", "/cameras/right_fork/semantic"}
                                          });

    std::map<std::string, std::string> camera_topics;
    if (!this->get_parameters<std::string>(params_prefix, camera_topics))
    {
        RCLCPP_ERROR(get_logger(), "Failed to get parameters, sensor topic names!");
        return;
    }

    mLeftRgbSub.subscribe(this, camera_topics.at("Body.Left"));
    mRightRgbSub.subscribe(this, camera_topics.at("Body.Right"));
    mLeftSemanticSub.subscribe(this, camera_topics.at("Body.LeftSemantic"));
    mRightSemanticSub.subscribe(this, camera_topics.at("Body.RightSemantic"));
    mLeftForkRgbSub.subscribe(this, camera_topics.at("Fork.Left"));
    mLeftForkSemanticSub.subscribe(this, camera_topics.at("Fork.LeftSemantic"));
    mRightForkRgbSub.subscribe(this, camera_topics.at("Fork.Right"));
    mRightForkSemanticSub.subscribe(this, camera_topics.at("Fork.RightSemantic"));
    mSynchronizer = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(8), mLeftRgbSub, mLeftSemanticSub,
                                                                                mRightRgbSub, mRightSemanticSub,
                                                                                mLeftForkRgbSub, mLeftForkSemanticSub,
                                                                                mRightForkRgbSub, mRightForkSemanticSub);

    // 设置更小的时间容差（单位：秒）
    mSynchronizer->setMaxIntervalDuration(rclcpp::Duration(0, 10 * 100000)); // 10ms 容差
    mSynchronizer->registerCallback(std::bind(&Preprocess::imgsHandler, this,
                                              std::placeholders::_1, std::placeholders::_2,
                                              std::placeholders::_3, std::placeholders::_4,
                                              std::placeholders::_5, std::placeholders::_6,
                                              std::placeholders::_7, std::placeholders::_8));
}

void Preprocess::initPublishers()
{
    mBevMapPub = create_publisher<sensor_msgs::msg::Image>("/bev_map", rclcpp::SensorDataQoS());
}

void Preprocess::imgsHandler(const ImgMsg::ConstSharedPtr& left_rgb_msg, const ImgMsg::ConstSharedPtr& left_semantic_msg,
                             const ImgMsg::ConstSharedPtr& right_rgb_msg, const ImgMsg::ConstSharedPtr& right_semantic_msg,
                             const ImgMsg::ConstSharedPtr& left_fork_rgb_msg, const ImgMsg::ConstSharedPtr& left_fork_semantic_msg,
                             const ImgMsg::ConstSharedPtr& right_fork_rgb_msg, const ImgMsg::ConstSharedPtr& right_fork_semantic_msg)
{
    /* Get the pose of the forks */
    // Eigen::Isometry3f T_body2fork;
    // try
    // {
    //     // This returns the pose of 'fork' in 'body' coordinates
    //     const geometry_msgs::msg::TransformStamped tf_body2fork = mTfBuffer->lookupTransform("LOLA", "fork", tf2::TimePointZero);
    //     T_body2fork = tf2::transformToEigen(tf_body2fork).cast<float>();
    // }
    // catch (const tf2::TransformException& ex)
    // {
    //     RCLCPP_ERROR(this->get_logger(), "Could not transform fork to body: %s", ex.what());
    //     return;
    // }

    ImgSet img_set{};
    try
    {
        const auto left_rgb_ptr = cv_bridge::toCvShare(left_rgb_msg, sensor_msgs::image_encodings::BGRA8);
        const auto right_rgb_ptr = cv_bridge::toCvShare(right_rgb_msg, sensor_msgs::image_encodings::BGRA8);
        const auto left_semantic_ptr = cv_bridge::toCvShare(left_semantic_msg, sensor_msgs::image_encodings::TYPE_32SC1);
        const auto right_semantic_ptr = cv_bridge::toCvShare(right_semantic_msg, sensor_msgs::image_encodings::TYPE_32SC1);

        const auto left_fork_rgb_ptr = cv_bridge::toCvShare(left_fork_rgb_msg, sensor_msgs::image_encodings::BGRA8);
        const auto right_fork_rgb_ptr = cv_bridge::toCvShare(right_fork_rgb_msg, sensor_msgs::image_encodings::BGRA8);
        const auto left_fork_semantic_ptr = cv_bridge::toCvShare(left_fork_semantic_msg, sensor_msgs::image_encodings::TYPE_32SC1);
        const auto right_fork_semantic_ptr = cv_bridge::toCvShare(right_fork_semantic_msg, sensor_msgs::image_encodings::TYPE_32SC1);

        // img_set.T_body2fork = T_body2fork;
        cv::cvtColor(left_rgb_ptr->image, img_set.left_rgb, cv::COLOR_BGRA2RGB);
        cv::cvtColor(right_rgb_ptr->image, img_set.right_rgb, cv::COLOR_BGRA2RGB);

        cv::cvtColor(left_fork_rgb_ptr->image, img_set.left_fork_rgb, cv::COLOR_BGRA2RGB);
        cv::cvtColor(right_fork_rgb_ptr->image, img_set.right_fork_rgb, cv::COLOR_BGRA2RGB);

        // img_set.left_rgb = left_rgb_ptr->image.clone();
        // img_set.right_rgb = right_rgb_ptr->image.clone();
        left_semantic_ptr->image.convertTo(img_set.left_semantic, CV_8UC1);
        right_semantic_ptr->image.convertTo(img_set.right_semantic, CV_8UC1);

        left_fork_semantic_ptr->image.convertTo(img_set.left_fork_semantic, CV_8UC1);
        right_fork_semantic_ptr->image.convertTo(img_set.right_fork_semantic, CV_8UC1);
    }
    catch (const cv_bridge::Exception& ex)
    {
        RCLCPP_WARN(get_logger(), "Failed to convert synchronized camera images: %s", ex.what());
        return;
    }

    pushInBuffer(std::move(img_set));
    mCameraFrameReceived.notify_one();
}

constexpr double DEG2RAD = CV_PI / 180.0;

cv::Matx33d rotX(const double deg)
{
    const double a = deg * DEG2RAD;
    const double c = std::cos(a);
    const double s = std::sin(a);

    return {
                1, 0, 0,
                0, c, -s,
                0, s, c
            };
}

cv::Matx33d rotY(const double deg)
{
    const double a = deg * DEG2RAD;
    const double c = std::cos(a);
    const double s = std::sin(a);

    return cv::Matx33d(
        c, 0, s,
        0, 1, 0,
        -s, 0, c
    );
}

cv::Matx33d rotZ(const double deg)
{
    const double a = deg * DEG2RAD;
    const double c = std::cos(a);
    const double s = std::sin(a);

    return cv::Matx33d(
        c, -s, 0,
        s, c, 0,
        0, 0, 1
    );
}

void Preprocess::workerLoop()
{
    // cv::Mat K = (cv::Mat_<double>(3, 3) <<
    //              fx, 0.0, cx,
    //              0.0, fy, cy,
    //              0.0, 0.0, 1.0);
    cv::Matx33d K(fx, 0.0, cx,
                  0.0, fy, cy,
                  0.0, 0.0, 1.0);

    BevConfig config;
    config.resolution = 0.01;
    config.x_max = 0.6;
    config.x_min = -3.5;
    config.y_max = 1.5;
    config.y_min = -1.5;

    const auto Rwc = rotZ(90.0) * rotY(0.0) * rotX(-105.0);
    const cv::Vec3d t_wc_l(1.25, -0.5, 1.2);
    const cv::Vec3d t_wc_r(1.25, 0.5, 1.2);

    // const cv::Vec3d t_wc_lfork(-1.39, -0.2, 0.18446156519147458);
    // const cv::Vec3d t_wc_rfork(-1.39, 0.2, 0.18446156519147458);

    const cv::Vec3d t_wc_lfork(-1.39, -0.2, 0.2);
    const cv::Vec3d t_wc_rfork(-1.39, 0.2, 0.2);

    const cv::Matx44d Twc_l(Rwc(0, 0), Rwc(0, 1), Rwc(0, 2), t_wc_l(0),
                            Rwc(1, 0), Rwc(1, 1), Rwc(1, 2), t_wc_l(1),
                            Rwc(2, 0), Rwc(2, 1), Rwc(2, 2), t_wc_l(2),
                            0.0, 0.0, 0.0, 1.0);
    const cv::Matx44d Twc_r(Rwc(0, 0), Rwc(0, 1), Rwc(0, 2), t_wc_r(0),
                            Rwc(1, 0), Rwc(1, 1), Rwc(1, 2), t_wc_r(1),
                            Rwc(2, 0), Rwc(2, 1), Rwc(2, 2), t_wc_r(2),
                            0.0, 0.0, 0.0, 1.0);

    const cv::Matx44d Twc_lfork(Rwc(0, 0), Rwc(0, 1), Rwc(0, 2), t_wc_lfork(0),
                                Rwc(1, 0), Rwc(1, 1), Rwc(1, 2), t_wc_lfork(1),
                                Rwc(2, 0), Rwc(2, 1), Rwc(2, 2), t_wc_lfork(2),
                                0.0, 0.0, 0.0, 1.0);
    const cv::Matx44d Twc_rfork(Rwc(0, 0), Rwc(0, 1), Rwc(0, 2), t_wc_rfork(0),
                                Rwc(1, 0), Rwc(1, 1), Rwc(1, 2), t_wc_rfork(1),
                                Rwc(2, 0), Rwc(2, 1), Rwc(2, 2), t_wc_rfork(2),
                                0.0, 0.0, 0.0, 1.0);

    const auto Tcw_l = Twc_l.inv();
    const auto Tcw_r = Twc_r.inv();

    const auto Tcw_lfork = Twc_lfork.inv();
    const auto Tcw_rfork = Twc_rfork.inv();

    const auto Rcw_l = Tcw_l.get_minor<3, 3>(0, 0);
    const auto Rcw_r = Tcw_r.get_minor<3, 3>(0, 0);
    const cv::Vec3d t_cw_l(Tcw_l(0, 3), Tcw_l(1, 3), Tcw_l(2, 3));
    const cv::Vec3d t_cw_r(Tcw_r(0, 3), Tcw_r(1, 3), Tcw_r(2, 3));

    const auto Rcw_lfork = Tcw_lfork.get_minor<3, 3>(0, 0);
    const auto Rcw_rfork = Tcw_rfork.get_minor<3, 3>(0, 0);
    const cv::Vec3d t_cw_lfork(Tcw_lfork(0, 3), Tcw_lfork(1, 3), Tcw_lfork(2, 3));
    const cv::Vec3d t_cw_rfork(Tcw_rfork(0, 3), Tcw_rfork(1, 3), Tcw_rfork(2, 3));

    std::vector<CameraModel> cameras(4); // left_camera, right_camera, left_fork_camera, right_fork_camera;
    cameras[0].K = K;
    cameras[0].Rcw = Rcw_l;
    cameras[0].tcw = t_cw_l;

    cameras[1].K = K;
    cameras[1].Rcw = Rcw_r;
    cameras[1].tcw = t_cw_r;

    cameras[2].K = K;
    cameras[2].Rcw = Rcw_lfork;
    cameras[2].tcw = t_cw_lfork;

    cameras[3].K = K;
    cameras[3].Rcw = Rcw_rfork;
    cameras[3].tcw = t_cw_rfork;

    while (rclcpp::ok())
    {
        ImgSet imgs;
        //
        {
            std::unique_lock lock(mImgBufferMutex);
            mCameraFrameReceived.wait(lock, [this]() -> bool { return !mImgsBuffer.empty() || mIsShutdown; });
            if (mIsShutdown)
            {
                break;
            }
            imgs = std::move(mImgsBuffer.front());
            mImgsBuffer.pop();
        }

        cv::Mat left_floor = cv::Mat::zeros(imgs.left_rgb.size(), CV_8UC1);
        cv::Mat right_floor = cv::Mat::zeros(imgs.right_rgb.size(), CV_8UC1);
        // const cv::Mat a = imgs.left_semantic * 30;
        constexpr uchar floor_label = 7;
        left_floor.setTo(255, imgs.left_semantic == floor_label);
        right_floor.setTo(255, imgs.right_semantic == floor_label);


        cv::Mat left_fork_floor = cv::Mat::zeros(imgs.left_fork_rgb.size(), CV_8UC1);
        cv::Mat right_fork_floor = cv::Mat::zeros(imgs.right_fork_rgb.size(), CV_8UC1);
        left_fork_floor.setTo(255, imgs.left_fork_semantic == floor_label);
        right_fork_floor.setTo(255, imgs.right_fork_semantic == floor_label);

        imgs.left_rgb.setTo(cv::Vec3b(0, 0, 0), left_floor == 0);
        imgs.right_rgb.setTo(cv::Vec3b(0, 0, 0), right_floor == 0);

        imgs.left_fork_rgb.setTo(cv::Vec3b(0, 0, 0), left_fork_floor == 0);
        imgs.right_fork_rgb.setTo(cv::Vec3b(0, 0, 0), right_fork_floor == 0);

        cameras[0].image = imgs.left_rgb;
        cameras[1].image = imgs.right_rgb;

        cameras[2].image = imgs.left_fork_rgb;
        cameras[3].image = imgs.right_fork_rgb;

        // const cv::Mat bev = bevFusionV2(cameras, config);
        const auto [bev, bev_binary] = bevFusionV2(cameras, config);

        cv_bridge::CvImage img_bridge;
        img_bridge.header.stamp = this->now(); // Optional: set a timestamp
        img_bridge.header.frame_id = "camera_frame";
        img_bridge.encoding = sensor_msgs::image_encodings::TYPE_8UC1; // e.g., "bgr8"
        img_bridge.image = bev_binary;
        auto ros_image = std::make_unique<sensor_msgs::msg::Image>();
        // sensor_msgs::msg::Image ros_image;
        img_bridge.toImageMsg(*ros_image);
        mBevMapPub->publish(std::move(ros_image));

        const auto feasible_region = computeFeasibleRegion(bev_binary, cv::Size(80, 60));
        // const auto dist_field = computeDistanceField(bev_binary);
        // const auto pose = findBestPlacement(feasible_region, dist_field);

        std::vector<cv::Point> feasible_points;
        cv::findNonZero(feasible_region, feasible_points);
        if (feasible_points.empty())
        {
            RCLCPP_DEBUG(get_logger(), "No feasible BEV placement region found for current frame.");
            continue;
        }
        std::ranges::sort(feasible_points, std::less{}, &cv::Point::x);
        auto max_it = std::ranges::max_element(feasible_points.begin(), feasible_points.begin() + std::min(8uz, feasible_points.size()),
                                               {}, &cv::Point::y);

        cv::Mat debug_img;
        cv::cvtColor(bev_binary, debug_img, cv::COLOR_GRAY2BGR);
        cv::rectangle(debug_img, cv::Rect(max_it->x - 40, max_it->y, 80, 60), cv::Scalar(0, 255, 0), 2);
    }
}
