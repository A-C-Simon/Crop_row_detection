/***************************************************************************************/
/* Paper: Visual-Servoing based Navigation for Monitoring Row-Crop Fields              */
/*    Alireza Ahmadi, Lorenzo Nardi, Nived Chebrolu, Chis McCool, Cyrill Stachniss     */
/*         All authors are with the University of Bonn, Germany                        */
/* maintainer: Alireza Ahmadi                                                          */
/*          (Alireza.Ahmadi@uni-bonn.de / http://alirezaahmadi.xyz)                    */
/***************************************************************************************/

#pragma once

#include "agribot_vs.h"
#include <rosgraph_msgs/msg/clock.hpp>

using namespace cv;
using namespace std;
using namespace Eigen;

namespace agribot_vs {
/**
 * @brief node handler class of VisualServoing application
 * 
 */
class AgribotVSNodeHandler {
 public:
  /**
   * @brief Construct a new Agribot V S Node Handler object
   * 
   * @param node_handler 
   */
  AgribotVSNodeHandler(rclcpp::Node::SharedPtr node_handler);
  /**
   * @brief Destroy the Agribot V S Node Handler object
   * 
   */
  virtual ~AgribotVSNodeHandler();
  /**
   * @brief gets the input camera (primary one) to extract crop-rows and visual featues
   * 
   * @param src is primary camera
   */
  void CropRow_Tracking(camera& src);

  void imuCallBack(const sensor_msgs::msg::Imu::ConstSharedPtr& msg);
  /**
   * @brief gets front camera's image
   * 
   * @param msg 
   */
  void imageFrontCalllBack(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
  /**
   * @brief gets rear camera's image
   * 
   * @param msg 
   */
  void imageBackCalllBack(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
  /**
   * @brief gets the robot odometry from base controller
   * 
   * @param msg 
   */
  void odomCallBack(const nav_msgs::msg::Odometry::ConstSharedPtr& msg);
  /**
   * @brief gets the poseof the robot in Lab from Mocap system
   * 
   * @param msg 
   */
  void amclPoseCallBack(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& msg);
  /**
   * @brief stops the robot fror given time
   * 
   * @param delay 
   */
  void StopForSec(float delay);
  /**
   * @brief publishes /cmd_vel topic to move the robot 
   * 
   * @param _in 
   */
  void publishVelocity(int _in=1);
  // void dynamicReconfig_callback(visual_crop_row_navigation_ros2::AgribotVSConfig &config, uint32_t level);

  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr Time_pub;
  AgribotVS agribotVS;

 private:

  int state, in_state;

  // ROS node handle.
  rclcpp::Node::SharedPtr nodeHandle_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_front_sub;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_back_sub;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr Mocap_sub;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr Odom_sub;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr IMU_sub;
  
  double mocap_roll, mocap_pitch, mocap_yaw;
  double imu_roll, imu_pitch, imu_yaw;
  rclcpp::Publisher<visual_crop_row_navigation_ros2::msg::VsMsg>::SharedPtr Log_pub;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr VSVelocityPub;
  

};
}  // namespace agribot_vs