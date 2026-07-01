#include "perception/nodes/Preprocess.h"
// #include "perception/nodes/PoseEstimation.h"

int main(const int argc, char** argv)
{
    rclcpp::init(argc, argv);
    const auto options = rclcpp::NodeOptions().use_intra_process_comms(true);
    const auto cloud_pub_node = std::make_shared<Preprocess>("bev_publish", options);
    // const auto target_pose_pub_node = std::make_shared<PoseEstimation>("target_publisher", options);
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(cloud_pub_node);
    // executor.add_node(target_pose_pub_node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
