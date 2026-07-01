#include "perception/nodes/Preprocess.h"
#include <cv_bridge/cv_bridge.hpp>
#include <Eigen/Geometry>
#include <algorithm>
#include "perception/tools/bev.h"

enum
{
    PALLET = 0,
    STORAGE_CAGE = 1,
    GOODS = 2
};

void Preprocess::initSubscritions()
{
    const std::string params_prefix = "TopicName.Sensor.Camera";
    this->declare_parameters<std::string>(params_prefix, {
                                              {"Body.Left", "/cameras/left/rgb"},
                                              {"Body.Right", "/cameras/right/rgb"}
                                          });

    std::map<std::string, std::string> camera_topics;
    if (!this->get_parameters<std::string>(params_prefix, camera_topics))
    {
        RCLCPP_ERROR(get_logger(), "Failed to get parameters, sensor topic names!");
    }

    // mLeftRgbSub.subscribe(this, camera_topics["Body.Left"]);
    // mRightRgbSub.subscribe(this, camera_topics["Body.Right"]);

    mLeftRgbSub.subscribe(this, "/cameras/left/rgb");
    mRightRgbSub.subscribe(this, "/cameras/right/rgb");
    mLeftSemanticSub.subscribe(this, "/cameras/left/semantic");
    mRightSemanticSub.subscribe(this, "/cameras/right/semantic");
    mSynchronizer = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(4), mLeftRgbSub, mLeftSemanticSub,
                                                                                mRightRgbSub, mRightSemanticSub);

    // 设置更小的时间容差（单位：秒）
    mSynchronizer->setMaxIntervalDuration(rclcpp::Duration(0, 10 * 100000)); // 10ms 容差
    mSynchronizer->registerCallback(std::bind(&Preprocess::imgsHandler, this,
                                              std::placeholders::_1, std::placeholders::_2,
                                              std::placeholders::_3, std::placeholders::_4));
}

void Preprocess::initPublishers()
{
    mBevMapPub = create_publisher<sensor_msgs::msg::Image>("/bev_map", rclcpp::SensorDataQoS());
}

void Preprocess::imgsHandler(const ImgMsg::ConstSharedPtr& left_rgb_msg, const ImgMsg::ConstSharedPtr& left_semantic_msg,
                             const ImgMsg::ConstSharedPtr& right_rgb_msg, const ImgMsg::ConstSharedPtr& right_semantic_msg)
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

    const auto left_rgb_ptr = cv_bridge::toCvShare(left_rgb_msg, sensor_msgs::image_encodings::BGRA8);
    const auto right_rgb_ptr = cv_bridge::toCvShare(right_rgb_msg, sensor_msgs::image_encodings::BGRA8);

    const auto left_semantic_ptr = cv_bridge::toCvShare(left_semantic_msg, sensor_msgs::image_encodings::TYPE_32SC1);
    const auto right_semantic_ptr = cv_bridge::toCvShare(right_semantic_msg, sensor_msgs::image_encodings::TYPE_32SC1);

    ImgSet img_set{};
    // img_set.T_body2fork = T_body2fork;
    cv::cvtColor(left_rgb_ptr->image, img_set.left_rgb, cv::COLOR_BGRA2RGB);
    cv::cvtColor(right_rgb_ptr->image, img_set.right_rgb, cv::COLOR_BGRA2RGB);

    // img_set.left_rgb = left_rgb_ptr->image.clone();
    // img_set.right_rgb = right_rgb_ptr->image.clone();
    left_semantic_ptr->image.convertTo(img_set.left_semantic, CV_8UC1);
    //img_set.left_rgb *= 30;
    right_semantic_ptr->image.convertTo(img_set.right_semantic, CV_8UC1);
    //img_set.right_rgb *= 30;

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
    constexpr double x_min = -4.0;
    constexpr double x_max = 4.0;
    constexpr double y_min = -2.0;
    constexpr double y_max = 2.0;
    constexpr double resolution = 0.01; // 1 cm per BEV pixel

    // cv::Mat K = (cv::Mat_<double>(3, 3) <<
    //              fx, 0.0, cx,
    //              0.0, fy, cy,
    //              0.0, 0.0, 1.0);
    cv::Matx33d K(fx, 0.0, cx,
                  0.0, fy, cy,
                  0.0, 0.0, 1.0);

    BevConfig config;
    config.resolution = 0.01;
    // config.x_max = 3.0;
    // config.x_min = -3.0;
    // config.y_max = 1.5;
    // config.y_min = -1.5;

    config.x_max = 2.0;
    config.x_min = -4.0;
    config.y_max = 2.5;
    config.y_min = -2.5;

    // Rcw = [105, 0, -90]
    // Left: t_cw = [-1.25, 0.5, -1.3]
    const auto Rwc = rotZ(90.0) * rotY(0.0) * rotX(-105.0);
    // const cv::Vec3d t_cw_l(-1.25, 0.5, -1.3);
    const cv::Vec3d t_wc_l(1.25, -0.5, 1.3);
    // const cv::Vec3d t_cw_r(-1.25, -0.5, -1.3);
    const cv::Vec3d t_wc_r(1.25, 0.5, 1.3);

    const cv::Matx44d Twc_l(Rwc(0, 0), Rwc(0, 1), Rwc(0, 2), t_wc_l(0),
                            Rwc(1, 0), Rwc(1, 1), Rwc(1, 2), t_wc_l(1),
                            Rwc(2, 0), Rwc(2, 1), Rwc(2, 2), t_wc_l(2),
                            0.0, 0.0, 0.0, 1.0);

    const cv::Matx44d Twc_r(Rwc(0, 0), Rwc(0, 1), Rwc(0, 2), t_wc_r(0),
                            Rwc(1, 0), Rwc(1, 1), Rwc(1, 2), t_wc_r(1),
                            Rwc(2, 0), Rwc(2, 1), Rwc(2, 2), t_wc_r(2),
                            0.0, 0.0, 0.0, 1.0);

    const auto Tcw_l = Twc_l.inv();
    const auto Tcw_r = Twc_r.inv();

    const auto Rcw_l = Tcw_l.get_minor<3, 3>(0, 0);
    const auto Rcw_r = Tcw_r.get_minor<3, 3>(0, 0);
    const cv::Vec3d t_cw_l(Tcw_l(0, 3), Tcw_l(1, 3), Tcw_l(2, 3));
    const cv::Vec3d t_cw_r(Tcw_r(0, 3), Tcw_r(1, 3), Tcw_r(2, 3));
    // const cv::Vec3d t_cw_l(1.25, -0.5, 1.3);
    // const cv::Vec3d t_cw_r(1.25, 0.5, 1.3);


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
        const cv::Mat a = imgs.left_semantic * 30;
        constexpr uchar floor_label = 8;
        left_floor.setTo(255, imgs.left_semantic == floor_label);
        cv::Mat right_floor = cv::Mat::zeros(imgs.right_rgb.size(), CV_8UC1);
        right_floor.setTo(255, imgs.right_semantic == floor_label);

        // const auto left_bev = groundMaskToBev(imgs.left_rgb, left_floor, K, Rcw, t_cw_l, config);
        // const auto right_bev = groundMaskToBev(imgs.right_rgb, right_floor, K, Rcw, t_cw_r, config);

        const auto left_bev = ipmInverseMapping(imgs.left_rgb, left_floor, K, Rcw_l, t_cw_l, config);
        const auto right_bev = ipmInverseMapping(imgs.right_rgb, right_floor, K, Rcw_r, t_cw_r, config);

        cv::Mat bev;
        cv::addWeighted(left_bev.bev_image, 1.0, right_bev.bev_image, 1.0, 0, bev);
        // imgs.left_rgb.setTo(cv::Scalar(0, 0, 0), left_floor == 0);
        // imgs.right_rgb.setTo(cv::Scalar(0, 0, 0), right_floor == 0);
        //
        // auto left_bev = getBev(imgs.left_rgb, K, Rcw, t_cw_l, config);
        // auto right_bev = getBev(imgs.right_rgb, K, Rcw, t_cw_r, config);

        // auto left_bev = getBev(imgs.left_rgb, K, Rwc, t_wc_l, config);
        // auto right_bev = getBev(imgs.right_rgb, K, Rwc, t_wc_r, config);
    }
}
