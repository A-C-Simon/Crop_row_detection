#!/usr/bin/env python3
"""Topic/QoS bridge between the Gazebo rover and the ExG C++ stack.

The agribot_vs node subscribes with default (Reliable) QoS and fixed topic
names, while Gazebo Classic publishes the camera BestEffort and odometry as
/odom:

  /camera/image_raw (BestEffort) -> /front/rgb/image_raw (Reliable)
  /odom              (as-is)     -> /odometry/raw       (republished)

IMU and AMCL inputs of the C++ node are left unconnected; it tolerates them
as zeros (level ground, no headland logic in the furrow).
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry


def _best_effort():
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      history=HistoryPolicy.KEEP_LAST, depth=5)


class Bridge(Node):
    def __init__(self):
        super().__init__("exgsim_bridge")
        self.img_pub = self.create_publisher(Image, "/front/rgb/image_raw", 10)
        self.odom_pub = self.create_publisher(Odometry, "/odometry/raw", 10)
        self.create_subscription(Image, "/camera/image_raw",
                                 self.img_pub.publish, _best_effort())
        self.create_subscription(Odometry, "/odom",
                                 self.odom_pub.publish, 10)
        self.get_logger().info("bridging /camera/image_raw -> "
                               "/front/rgb/image_raw, /odom -> /odometry/raw")


def main(args=None):
    rclpy.init(args=args)
    node = Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
