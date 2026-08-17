#include "perception/nodes/LidarCameraFusion.h"

int main(const int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarCameraFusionNode>());
    rclcpp::shutdown();
    return 0;
}

