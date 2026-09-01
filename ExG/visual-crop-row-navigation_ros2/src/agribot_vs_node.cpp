/***************************************************************************************/
/* Paper: Visual-Servoing based Navigation for Monitoring Row-Crop Fields              */
/*    Alireza Ahmadi, Lorenzo Nardi, Nived Chebrolu, Chis McCool, Cyrill Stachniss     */
/*         All authors are with the University of Bonn, Germany                        */
/* maintainer: Alireza Ahmadi                                                          */
/*          (Alireza.Ahmadi@uni-bonn.de / http://alirezaahmadi.xyz)                    */
/***************************************************************************************/
#include <rclcpp/rclcpp.hpp>
#include "agribot_vs_nodehandler.h"
#include "agribot_vs.h"

#include <sstream>
#include <time.h>

int main(int argc, char** argv) {
  // initialize node
  rclcpp::init(argc, argv);

  // node handler
  rclcpp::Node::SharedPtr nodeHandle = std::make_shared<rclcpp::Node>("agribot_vs");
  agribot_vs::AgribotVSNodeHandler vs_NodeH(nodeHandle);
  rclcpp::Rate loop_rate(vs_NodeH.agribotVS.fps);

  if(vs_NodeH.agribotVS.mask_tune)cout << "Mask Tune Mode ..." << endl;
  if(vs_NodeH.agribotVS.single_camera_mode)cout << "Single Camera Mode (Front Camera will only be used..)" << endl;

  agribot_vs::camera *I_primary,*I_secondary;
  if(vs_NodeH.agribotVS.camera_ID == 1){
    I_primary = &vs_NodeH.agribotVS.front_cam;
    I_secondary = &vs_NodeH.agribotVS.back_cam;
  }else {
    I_primary = &vs_NodeH.agribotVS.back_cam;
    I_secondary = &vs_NodeH.agribotVS.front_cam;
  }
  vs_NodeH.agribotVS.initialize_neigbourhood(*I_primary);
  vs_NodeH.agribotVS.initialize_neigbourhood(*I_secondary);
  int cnt =0;
  
  while(rclcpp::ok()){

    if(cnt < vs_NodeH.agribotVS.max_row_num){

      if(vs_NodeH.agribotVS.single_camera_mode){
        I_secondary = I_primary;
      }
      if(!vs_NodeH.agribotVS.mask_tune){
        vs_NodeH.agribotVS.switching_controller(*I_primary, *I_secondary, vs_NodeH.agribotVS.min_points_switch);

        if(vs_NodeH.agribotVS.camera_ID == 1){
          I_primary = &vs_NodeH.agribotVS.front_cam;
          I_secondary = &vs_NodeH.agribotVS.back_cam;
        }else {
          I_primary = &vs_NodeH.agribotVS.back_cam;
          I_secondary = &vs_NodeH.agribotVS.front_cam;
        }

        if(!I_primary->lines.empty()){
          vs_NodeH.agribotVS.compute_feature_point(*I_primary);
          vs_NodeH.agribotVS.Controller(*I_primary,*I_secondary);
        }

        if(!I_primary->image.empty()){
          vs_NodeH.agribotVS.draw_neighbourhood(*I_primary);
          vs_NodeH.agribotVS.draw_features(*I_primary, vs_NodeH.agribotVS.F_des, cv::Scalar(0, 255, 0));
          vs_NodeH.agribotVS.draw_features(*I_primary, vs_NodeH.agribotVS.F, cv::Scalar(0, 0, 255));

          // draw plant centers in image (in neighbourhood)
          for(size_t i = 0; i < I_primary->nh_points.size(); i++){
            cv::circle(I_primary->image, Point(I_primary->nh_points[i].x,I_primary->nh_points[i].y),5, Scalar(0, 204, 255), cv::FILLED, 8,0);
          }

          Mat des_comp;
          cv::resize(I_primary->image, des_comp, cv::Size(), vs_NodeH.agribotVS.Scale, vs_NodeH.agribotVS.Scale);
          imshow("Cameras", des_comp);
          waitKey(1);
        }else{
          cout << "Is Image empty? Camera-Primary: " << I_primary->image.empty() <<  " , Camera-Secondary "<< I_secondary->image.empty() << endl;
        }
      }

      if(I_primary->points.size() == 0){
        vs_NodeH.publishVelocity(0);
      }else{
        vs_NodeH.publishVelocity();
      }
      
      rosgraph_msgs::msg::Clock curr_time_msg;
      curr_time_msg.clock = rclcpp::Clock(RCL_STEADY_TIME).now();      
      vs_NodeH.Time_pub->publish(curr_time_msg);
    }
    
    if(cnt < 1000)cnt++;
    rclcpp::spin_some(nodeHandle);
    loop_rate.sleep();
  }
 
  rclcpp::shutdown();
  return 0;
}