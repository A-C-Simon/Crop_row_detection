#!/usr/bin/env python3
"""Grab a single image from a ROS2 topic and save it to /tmp/grabbed.png.

    python3 grab_img.py <topic>
"""
import sys
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class Grab(Node):
    def __init__(self, topic):
        super().__init__("grab")
        self.b = CvBridge()
        self.create_subscription(Image, topic, self.cb, 5)

    def cb(self, msg):
        img = self.b.imgmsg_to_cv2(msg, "bgr8")
        cv2.imwrite("/tmp/grabbed.png", img)
        self.get_logger().info(
            f"saved {img.shape} mean={img.mean(axis=(0, 1)).round(1)}")
        rclpy.shutdown()


def main():
    rclpy.init()
    topic = sys.argv[1] if len(sys.argv) > 1 else "/camera/image_raw"
    rclpy.spin(Grab(topic))


if __name__ == "__main__":
    main()
