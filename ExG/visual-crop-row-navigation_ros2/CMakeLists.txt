cmake_minimum_required(VERSION 3.8)
project(visual_crop_row_navigation_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/VsMsg.msg"
  DEPENDENCIES std_msgs
)

include_directories(include)

## Declare a cpp library
add_library(${PROJECT_NAME}_core STATIC src/agribot_vs.cpp)
target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)
ament_target_dependencies(${PROJECT_NAME}_core
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  cv_bridge
  tf2
  tf2_geometry_msgs
)
target_link_libraries(${PROJECT_NAME}_core ${OpenCV_LIBS} Eigen3::Eigen)
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_core "${cpp_typesupport_target}")

## Declare cpp executables
add_executable(agribot_vs_node
  src/agribot_vs_node.cpp
  src/agribot_vs_nodehandler.cpp
)
target_include_directories(agribot_vs_node PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
)
ament_target_dependencies(agribot_vs_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  rosgraph_msgs
  cv_bridge
  tf2
  tf2_geometry_msgs
)
target_link_libraries(agribot_vs_node
  ${PROJECT_NAME}_core
  ${OpenCV_LIBS}
  Eigen3::Eigen
)
target_link_libraries(agribot_vs_node "${cpp_typesupport_target}")

install(TARGETS agribot_vs_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch params
  DESTINATION share/${PROJECT_NAME}
)

ament_package()