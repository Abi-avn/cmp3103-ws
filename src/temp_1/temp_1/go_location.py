#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import time


class FullyAutoNav(Node):

    def __init__(self):
        super().__init__('fully_auto_nav')

        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        self.timer = self.create_timer(1.0, self.run)

        self.step = 0

        self.get_logger().info("🚀 Fully autonomous nav started")

    def run(self):

        # STEP 1 — Set initial pose (what RViz did)
        if self.step == 0:
            self.get_logger().info("📍 Setting initial pose...")

            pose = PoseWithCovarianceStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()

            # 🔥 SET THIS ONCE (your correct start position)
            pose.pose.pose.position.x = 0.0
            pose.pose.pose.position.y = 0.0
            pose.pose.pose.orientation.w = 1.0

            self.pose_pub.publish(pose)

            self.step = 1
            return

        # STEP 2 — wait for AMCL to stabilise
        elif self.step == 1:
            self.get_logger().info("⏳ Waiting for AMCL...")
            time.sleep(3)
            self.step = 2
            return

        # STEP 3 — send goal (what RViz did)
        elif self.step == 2:
            self.get_logger().info("🎯 Sending goal...")

            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.header.stamp = self.get_clock().now().to_msg()

            # 📍 YOUR PATCH LOCATION
            goal.pose.position.x = -0.78
            goal.pose.position.y = 1.02
            goal.pose.orientation.w = 1.0

            self.goal_pub.publish(goal)

            self.get_logger().info("✅ Goal sent!")

            self.step = 3
            return

        elif self.step == 3:
            self.timer.cancel()


def main():
    rclpy.init()
    node = FullyAutoNav()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()