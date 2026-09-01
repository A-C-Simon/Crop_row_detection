/***************************************************************************************/
/* Paper: Visual-Servoing based Navigation for Monitoring Row-Crop Fields              */
/*    Alireza Ahmadi, Lorenzo Nardi, Nived Chebrolu, Chis McCool, Cyrill Stachniss     */
/*         All authors are with the University of Bonn, Germany                        */
/* maintainer: Alireza Ahmadi                                                          */
/*          (Alireza.Ahmadi@uni-bonn.de / http://alirezaahmadi.xyz)                    */
/***************************************************************************************/

#include "agribot_vs_nodehandler.h"
#include "agribot_vs.h"
#include <time.h>
#include <functional>
#include <chrono>
#include <thread>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace agribot_vs {

AgribotVSNodeHandler::AgribotVSNodeHandler(rclcpp::Node::SharedPtr nodeHandle): nodeHandle_(nodeHandle){
  RCLCPP_ERROR(nodeHandle_->get_logger(), "Visual Servoing core is running...");
  if (!agribotVS.readRUNParmas(nodeHandle_)) {
     RCLCPP_ERROR(nodeHandle_->get_logger(), "Could not read parameters.");
     rclcpp::shutdown();
  }

  // Subscribers
  image_front_sub = nodeHandle_->create_subscription<sensor_msgs::msg::Image>("/front/rgb/image_raw", 2, std::bind(&AgribotVSNodeHandler::imageFrontCalllBack,this,std::placeholders::_1));
  image_back_sub = nodeHandle_->create_subscription<sensor_msgs::msg::Image>("/back/rgb/image_raw", 2, std::bind(&AgribotVSNodeHandler::imageBackCalllBack,this,std::placeholders::_1));
  Mocap_sub = nodeHandle_->create_subscription<geometry_msgs::msg::PoseStamped>("/amcl_pose", 1, std::bind(&AgribotVSNodeHandler::amclPoseCallBack,this,std::placeholders::_1));
  Odom_sub = nodeHandle_->create_subscription<nav_msgs::msg::Odometry>("/odometry/raw", 10, std::bind(&AgribotVSNodeHandler::odomCallBack,this,std::placeholders::_1));
  IMU_sub = nodeHandle_->create_subscription<sensor_msgs::msg::Imu>("/imu/data", 1000, std::bind(&AgribotVSNodeHandler::imuCallBack,this,std::placeholders::_1));
  
  // Publishers
  Time_pub = nodeHandle_->create_publisher<rosgraph_msgs::msg::Clock>("/clock", 10);
  VSVelocityPub = nodeHandle_->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
  Log_pub = nodeHandle_->create_publisher<visual_crop_row_navigation_ros2::msg::VsMsg>("/vs_msg", 10);

  agribotVS.VelocityMsg.linear.x =0.0;
  agribotVS.VelocityMsg.angular.z =0.0;

  state = 0;
  in_state = 0;
}
AgribotVSNodeHandler::~AgribotVSNodeHandler() {
}
void AgribotVSNodeHandler::CropRow_Tracking(camera& src){
    // finding contour from image baed on crops in rows
    src.contours = agribotVS.CropRowFeatures(src);
    if(!agribotVS.mask_tune || src.contours.size() != 0){

      src.points = agribotVS.getContureCenters(src.image, src.contours);
      
      src.nh_points = agribotVS.filterContures(src.image, src.contours);

      agribotVS.is_in_neigbourhood(src);

      src.lines =  agribotVS.FitLineOnContures(src.image, src.nh_points);
    }else{
      cout << "Numner of contures: " << src.contours.size() << endl;
      publishVelocity(0);
    }
}
void AgribotVSNodeHandler::imageFrontCalllBack(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
  try {
    agribotVS.front_cam.image = cv_bridge::toCvCopy(msg, "bgr8")->image;
    CropRow_Tracking(agribotVS.front_cam);
    
    string str;
    stringstream stream,stream1,stream2;
    stream << agribotVS.front_cam.points.size(); 
    stream1 << agribotVS.front_cam.nh_points.size(); 
    stream2 << agribotVS.camera_ID;
    str = "Number of Points: " + stream.str() + " nh_points: " + stream1.str() + " Cam ID: " + stream2.str();
    cv::putText(agribotVS.front_cam.image, str, cv::Point(40, 20),  // Coordinates
                cv::FONT_HERSHEY_COMPLEX_SMALL,       // Font
                0.75,                                 // Scale. 2.0 = 2x bigger
                cv::Scalar(0, 0, 255),                // BGR Color
                1);                                   // Line Thickness (Optional)

  } catch (cv_bridge::Exception& e) {
    RCLCPP_ERROR(nodeHandle_->get_logger(), "Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}
void AgribotVSNodeHandler::imageBackCalllBack(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
  try {
    agribotVS.back_cam.image = cv_bridge::toCvCopy(msg, "bgr8")->image;
    CropRow_Tracking(agribotVS.back_cam);

    string str;
    stringstream stream,stream1,stream2;
    stream << agribotVS.back_cam.points.size(); 
    stream1 << agribotVS.back_cam.nh_points.size(); 
    stream2 << agribotVS.camera_ID;
    str = "Number of Points: " + stream.str() + " nh_points: " + stream1.str() + " Cam ID: " + stream2.str();
    cv::putText(agribotVS.back_cam.image, str, cv::Point(40, 20),  // Coordinates
                cv::FONT_HERSHEY_COMPLEX_SMALL,       // Font
                0.75,                                 // Scale. 2.0 = 2x bigger
                cv::Scalar(0, 0, 255),                // BGR Color
                1);                                   // Line Thickness (Optional)

  } catch (cv_bridge::Exception& e) {
    RCLCPP_ERROR(nodeHandle_->get_logger(), "Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}
void AgribotVSNodeHandler::imuCallBack(const sensor_msgs::msg::Imu::ConstSharedPtr& msg){
  tf2::Quaternion quat;
  tf2::fromMsg(msg->orientation, quat);
  tf2::Matrix3x3 m(quat);
  m.getRPY(imu_roll, imu_pitch, imu_yaw);
}
void AgribotVSNodeHandler::amclPoseCallBack(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& msg) {
  tf2::Quaternion q(msg->pose.orientation.x, msg->pose.orientation.y,
                   msg->pose.orientation.z, msg->pose.orientation.w);
  tf2::Matrix3x3 m(q);
  m.getRPY(mocap_roll, mocap_pitch, mocap_yaw);
  agribotVS.RotationVec[0] = 0;    
  agribotVS.RotationVec[1] = 0;    
  agribotVS.RotationVec[2] = mocap_yaw;  
  agribotVS.TransVec[0] = msg->pose.position.x;
  agribotVS.TransVec[1] = msg->pose.position.y;
  agribotVS.TransVec[2] = msg->pose.position.z;
}
void AgribotVSNodeHandler::odomCallBack(const nav_msgs::msg::Odometry::ConstSharedPtr& msg) {
  agribotVS.RobotPose[0] = msg->pose.pose.position.x;
  agribotVS.RobotPose[1] = msg->pose.pose.position.y;
  agribotVS.RobotPose[2] = msg->pose.pose.position.z;

  std::vector<double> Orineration = agribotVS.getEulerAngles(msg);

  agribotVS.RobotPose[3] = Orineration[0];  // x oreintation
  agribotVS.RobotPose[4] = Orineration[1];  // y oreintation
  agribotVS.RobotPose[5] = Orineration[2];  // z oreintation

  agribotVS.RobotLinearVelocities[0] = msg->twist.twist.linear.x;
  agribotVS.RobotLinearVelocities[1] = msg->twist.twist.linear.y;
  agribotVS.RobotLinearVelocities[2] = msg->twist.twist.linear.z;

  agribotVS.RobotAngularVelocities[0] = msg->twist.twist.angular.x;
  agribotVS.RobotAngularVelocities[1] = msg->twist.twist.angular.y;
  agribotVS.RobotAngularVelocities[2] = msg->twist.twist.angular.z;
}
void AgribotVSNodeHandler::StopForSec(float delay) {
  agribotVS.VelocityMsg.angular.z = 0.0;
  agribotVS.VelocityMsg.linear.x = 0.0;
  if(agribotVS.publish_cmd_vel)VSVelocityPub->publish(agribotVS.VelocityMsg);
  std::this_thread::sleep_for(std::chrono::duration<float>(delay));  // sleep for half a second
}
void AgribotVSNodeHandler::publishVelocity(int _in) {
  if(!agribotVS.publish_linear_vel)agribotVS.VelocityMsg.linear.x = 0.0;
  if(_in == 0){
    agribotVS.VelocityMsg.linear.x = 0.0;
    agribotVS.VelocityMsg.angular.z = 0.0;
    if(agribotVS.publish_cmd_vel)
    VSVelocityPub->publish(agribotVS.VelocityMsg);
  }else{
    if(agribotVS.publish_cmd_vel){
      VSVelocityPub->publish(agribotVS.VelocityMsg);
    }
  }
  Log_pub->publish(agribotVS.VSMsg);
}

}   // namespace agribot_vs