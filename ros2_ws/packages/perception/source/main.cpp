#include "perception/nodes/LidarCameraFusion.h"
#include "perception/nodes/TrailerLocalization.h"

int main(const int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrailerLocalization>("trailer_localization"));
    rclcpp::shutdown();
    // std::cout << "sizeof float is " << sizeof(float) << ", sizeof double is " << sizeof(double) << std::endl;
    return 0;
}

