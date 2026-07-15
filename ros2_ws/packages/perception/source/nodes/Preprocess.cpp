#include <cv_bridge/cv_bridge.hpp>
#include <algorithm>
#include <map>
#include <ranges>
#include <string>
#include "perception/tools/2d/feature_detect.h"
#include "perception/tools/bev.h"
#include "perception/nodes/Preprocess.h"

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
    mSlotVisPub = create_publisher<sensor_msgs::msg::Image>("/bev_map_slot_vis", rclcpp::SensorDataQoS());
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

static int rightmostX(const cv::Mat& src)
{
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(!src.empty());

    const int rows = src.rows;

    for (int x = src.cols - 1; x >= 0; --x)
    {
        for (int y = 0; y < rows; ++y)
        {
            if (src.ptr<uchar>(y)[x] != 0)
            {
                return x;
            }
        }
    }

    return 0;
}

/**
 * Calculate the pose of 
 */
static std::optional<Eigen::Vector3f> loadPoseEstimate(const cv::Mat& load_bev, cv::Size& load_size, std::vector<cv::Point>& bbox)
{
    constexpr int default_length = 80;
    const int rightmost_x = rightmostX(load_bev);

    cv::Mat cut_load_bev;
    if (rightmost_x > default_length)
    {
        cut_load_bev = cv::Mat::zeros(load_bev.size(), CV_8UC1);
        const int start_x = rightmost_x - default_length;
        load_bev.colRange(start_x, rightmost_x + 1).copyTo(cut_load_bev.colRange(start_x, rightmost_x + 1));
    }

    if (cut_load_bev.empty())
    {
        cut_load_bev = load_bev.clone();
    }

    cv::Mat cut_load_bev_right_edge;
    feature2d::detectEdge(cut_load_bev, cut_load_bev_right_edge);
    
    std::vector<cv::Point> points;
    cv::findNonZero(cut_load_bev_right_edge, points);
    if (points.empty())
    {
        return std::nullopt;
    }

    const cv::RotatedRect load_bbox = cv::minAreaRect(points);
    float angle_deg = load_bbox.angle;
    if (load_bbox.size.width < load_bbox.size.height)
    {
        angle_deg += 90.0F;
    }

    std::vector<cv::Point2f> corners(4);
    load_bbox.points(corners.data());

    bbox.resize(4);
    for(size_t i =0; i<4;++i)
    {
        bbox[i] = corners[i];
    }

    cv::Mat debug_img;
    cv::cvtColor(cut_load_bev_right_edge, debug_img, cv::COLOR_GRAY2BGR);
    const cv::Scalar rect_color(0, 0, 255);
    for (size_t i = 0; i < bbox.size(); ++i)
    {
        cv::line(debug_img, bbox[i], bbox[(i + 1) % bbox.size()], rect_color, 2);
    }

    std::ranges::sort(corners, std::less{}, &cv::Point2f::x);

    load_size.width = default_length;
    load_size.height = static_cast<int>(std::ceil(cv::norm(corners[2] - corners[3])));

    const cv::Point2f right_edge_mid = 0.5f * (corners[2] + corners[3]);

    return Eigen::Vector3f{right_edge_mid.x, right_edge_mid.y, angle_deg * static_cast<float>(DEG2RAD)};
}

static std::optional<std::pair<int, int>> slotPoseEstimate(const cv::Mat& free_space, const cv::Size& load_size)
{
    const auto feasible_region = computeFeasibleRegion(free_space, load_size);

    std::vector<cv::Point> feasible_points;
    cv::findNonZero(feasible_region, feasible_points);
    if (feasible_points.empty())
    {
        return std::nullopt;
    }
    std::ranges::sort(feasible_points, std::less{}, &cv::Point::x);
    const auto slot_position = std::ranges::max_element(feasible_points.begin(), feasible_points.begin() + std::min(20uz, feasible_points.size()),
                                           {}, &cv::Point::y);
    cv::Mat debug_img;
    cv::cvtColor(free_space, debug_img, cv::COLOR_GRAY2BGR);


    // cv::cvtColor(free_space, debug_img, cv::COLOR_GRAY2BGR);
    cv::rectangle(debug_img, cv::Rect(slot_position->x - load_size.width / 2, slot_position->y - load_size.height / 2
        , load_size.width, load_size.height), cv::Scalar(0, 255, 0), 2);
    return std::make_pair(slot_position->x + load_size.width / 2, slot_position->y);
}

void Preprocess::workerLoop()
{
    Eigen::Matrix3f K;
    K << fx, 0.0f, cx,
        0.0f, fy, cy,
        0.0f, 0.0f, 1.0f;

    BevConfig config;
    config.resolution = 0.01;
    config.x_max = 0.6;
    config.x_min = -3.5;
    config.y_max = 1.5;
    config.y_min = -1.5;

    Eigen::Isometry3f temp_pose{Eigen::Isometry3f::Identity()};
    const float roll = -105.0f * DEG2RAD;
    const float pitch = 0.0f;
    const float yaw = 90.0f * DEG2RAD;
    // const Eigen::Vector3f translation(1.25f, -0.5f, 1.2f);
    temp_pose.rotate(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
                     Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
                     Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX()));
    
    Eigen::Isometry3f Twc_l{temp_pose}, Twc_r{temp_pose}, Twc_lfork{temp_pose}, Twc_rfork{temp_pose};
    const Eigen::Vector3f t_wc_l(1.25, -0.5, 1.2);
    const Eigen::Vector3f t_wc_r(1.25, 0.5, 1.2);

    const Eigen::Vector3f t_wc_lfork(-1.39, -0.2, 0.18446156519147458);
    const Eigen::Vector3f t_wc_rfork(-1.39, 0.2, 0.18446156519147458);

    Twc_l.pretranslate(t_wc_l);
    Twc_r.pretranslate(t_wc_r);
    Twc_lfork.pretranslate(t_wc_lfork);
    Twc_rfork.pretranslate(t_wc_rfork);

    // Equivalent prerotate form:
    // temp_pose.prerotate(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
    //                     Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
    //                     Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX()));
    // Use pretranslate here when translation is expressed in the parent/world frame.
    // translate() would apply the offset in the already-rotated local frame.

    // const auto Rwc = rotZ(90.0) * rotY(0.0) * rotX(-105.0);
    

    // const cv::Matx44d Twc_l(Rwc(0, 0), Rwc(0, 1), Rwc(0, 2), t_wc_l(0),
    //                         Rwc(1, 0), Rwc(1, 1), Rwc(1, 2), t_wc_l(1),
    //                         Rwc(2, 0), Rwc(2, 1), Rwc(2, 2), t_wc_l(2),
    //                         0.0, 0.0, 0.0, 1.0);
    // const cv::Matx44d Twc_r(Rwc(0, 0), Rwc(0, 1), Rwc(0, 2), t_wc_r(0),
    //                         Rwc(1, 0), Rwc(1, 1), Rwc(1, 2), t_wc_r(1),
    //                         Rwc(2, 0), Rwc(2, 1), Rwc(2, 2), t_wc_r(2),
    //                         0.0, 0.0, 0.0, 1.0);

    // const cv::Matx44d Twc_lfork(Rwc(0, 0), Rwc(0, 1), Rwc(0, 2), t_wc_lfork(0),
    //                             Rwc(1, 0), Rwc(1, 1), Rwc(1, 2), t_wc_lfork(1),
    //                             Rwc(2, 0), Rwc(2, 1), Rwc(2, 2), t_wc_lfork(2),
    //                             0.0, 0.0, 0.0, 1.0);
    // const cv::Matx44d Twc_rfork(Rwc(0, 0), Rwc(0, 1), Rwc(0, 2), t_wc_rfork(0),
    //                             Rwc(1, 0), Rwc(1, 1), Rwc(1, 2), t_wc_rfork(1),
    //                             Rwc(2, 0), Rwc(2, 1), Rwc(2, 2), t_wc_rfork(2),
    //                             0.0, 0.0, 0.0, 1.0);

    // const auto Tcw_l = Twc_l.inv();
    // const auto Tcw_r = Twc_r.inv();

    // const auto Tcw_lfork = Twc_lfork.inv();
    // const auto Tcw_rfork = Twc_rfork.inv();

    // const auto Rcw_l = Tcw_l.get_minor<3, 3>(0, 0);
    // const auto Rcw_r = Tcw_r.get_minor<3, 3>(0, 0);
    // const cv::Vec3d t_cw_l(Tcw_l(0, 3), Tcw_l(1, 3), Tcw_l(2, 3));
    // const cv::Vec3d t_cw_r(Tcw_r(0, 3), Tcw_r(1, 3), Tcw_r(2, 3));

    // const auto Rcw_lfork = Tcw_lfork.get_minor<3, 3>(0, 0);
    // const auto Rcw_rfork = Tcw_rfork.get_minor<3, 3>(0, 0);
    // const cv::Vec3d t_cw_lfork(Tcw_lfork(0, 3), Tcw_lfork(1, 3), Tcw_lfork(2, 3));
    // const cv::Vec3d t_cw_rfork(Tcw_rfork(0, 3), Tcw_rfork(1, 3), Tcw_rfork(2, 3));

    const auto Tcw_l = Twc_l.inverse();
    const auto Tcw_r = Twc_r.inverse();
    const auto Tcw_lfork = Twc_lfork.inverse();
    const auto Tcw_rfork = Twc_rfork.inverse();
    
    const Eigen::Matrix3f Rcw_l = Tcw_l.rotation();
    const Eigen::Matrix3f Rcw_r = Tcw_r.rotation();
    const Eigen::Vector3f t_cw_l = Tcw_l.translation();
    const Eigen::Vector3f t_cw_r = Tcw_r.translation();

    const Eigen::Matrix3f Rcw_lfork = Tcw_lfork.rotation();
    const Eigen::Matrix3f Rcw_rfork = Tcw_rfork.rotation();
    const Eigen::Vector3f t_cw_lfork = Tcw_lfork.translation();
    const Eigen::Vector3f t_cw_rfork = Tcw_rfork.translation();


    std::vector<CameraModel> cameras(4); // left_camera, right_camera, left_fork_camera, right_fork_camera;
    cameras[2].K = K;
    cameras[2].Rcw = Rcw_l;
    cameras[2].tcw = t_cw_l;

    cameras[3].K = K;
    cameras[3].Rcw = Rcw_r;
    cameras[3].tcw = t_cw_r;

    cameras[0].K = K;
    cameras[0].Rcw = Rcw_lfork;
    cameras[0].tcw = t_cw_lfork;

    cameras[1].K = K;
    cameras[1].Rcw = Rcw_rfork;
    cameras[1].tcw = t_cw_rfork;

    constexpr int load_length = 80;
    while (rclcpp::ok())
    {
        ImgSet imgs;
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

        const cv::Mat a = imgs.left_semantic * 30;
        constexpr uchar floor_label = 6;
        constexpr uchar load_label = 9;
        const cv::Mat left_valid = (imgs.left_semantic == floor_label) | (imgs.left_semantic == load_label);
        imgs.left_semantic.setTo(0, left_valid == 0);
        const cv::Mat right_valid = (imgs.right_semantic == floor_label) | (imgs.right_semantic == load_label);
        imgs.right_semantic.setTo(0, right_valid == 0);

        /* Stage 1: Calculate load dimentions and pose */
        std::vector<CameraModel> camera_frames_load(2);
        camera_frames_load[0].K = K;
        camera_frames_load[0].Rcw = Rcw_l;
        camera_frames_load[0].tcw = t_cw_l;
        camera_frames_load[0].image = imgs.left_semantic == load_label;
        
        camera_frames_load[1].K = K;
        camera_frames_load[1].Rcw = Rcw_r;
        camera_frames_load[1].tcw = t_cw_r;
        camera_frames_load[1].image = imgs.right_semantic == load_label;

        const cv::Mat load_bev = bevFusionBina(camera_frames_load, config);
        cv::Size load_dimensions;
        std::vector<cv::Point> load_bbox;
        const auto load_estimate_result = loadPoseEstimate(load_bev, load_dimensions, load_bbox);
        if (!load_estimate_result.has_value())
        {
            RCLCPP_WARN(get_logger(), "Failed to calculate load pose!");
            continue; 
        }

        /* Stage 2: Calculate free space slot pose */
        std::vector<CameraModel> camera_frames_floor(4);
        camera_frames_floor[2].K = K;
        camera_frames_floor[2].Rcw = Rcw_l;
        camera_frames_floor[2].tcw = t_cw_l;
        camera_frames_floor[2].image = imgs.left_semantic == floor_label;
        
        camera_frames_floor[3].K = K;
        camera_frames_floor[3].Rcw = Rcw_r;
        camera_frames_floor[3].tcw = t_cw_r;
        camera_frames_floor[3].image = imgs.right_semantic == floor_label;
        
        camera_frames_floor[0].K = K;
        camera_frames_floor[0].Rcw = Rcw_lfork;
        camera_frames_floor[0].tcw = t_cw_lfork;
        camera_frames_floor[0].image = imgs.left_fork_semantic == floor_label;
        
        camera_frames_floor[1].K = K;
        camera_frames_floor[1].Rcw = Rcw_rfork;
        camera_frames_floor[1].tcw = t_cw_rfork;
        camera_frames_floor[1].image = imgs.right_fork_semantic == floor_label;

        cv::Mat free_space_bev = bevFusionBina(camera_frames_floor, config);
        cv::Mat bev_visualizetion;
        cv::cvtColor(free_space_bev, bev_visualizetion, cv::COLOR_GRAY2RGB);

        free_space_bev.colRange(cv::Range(load_estimate_result.value()[0], free_space_bev.cols)).setTo(0);

        if(const auto estimate_result = slotPoseEstimate(free_space_bev, load_dimensions); estimate_result.has_value())
        {
            cv::fillPoly(bev_visualizetion, std::vector<std::vector<cv::Point>>{load_bbox}, cv::Scalar(0, 0, 255), cv::LINE_AA);
            visualizeSlot(bev_visualizetion, estimate_result.value(), load_dimensions);

            cv_bridge::CvImage img_bridge;
            img_bridge.header.stamp = this->now(); // Optional: set a timestamp
            img_bridge.header.frame_id = "camera_frame";
            img_bridge.encoding = sensor_msgs::image_encodings::RGB8; // e.g., "bgr8"
            img_bridge.image = bev_visualizetion;
            auto ros_image = std::make_unique<sensor_msgs::msg::Image>();
            img_bridge.toImageMsg(*ros_image);
            mSlotVisPub->publish(std::move(ros_image));
        }
        else
        {
            RCLCPP_WARN(get_logger(), "No feasible BEV placement region found for current frame!");
        }
    }
}
