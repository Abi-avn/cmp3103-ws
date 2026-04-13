#!/usr/bin/env python3

import rclpy, time, cv2, numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import Image, LaserScan
from action_msgs.msg import GoalStatusArray
from cv_bridge import CvBridge


class TidyBotNav2(Node):

    def __init__(self):
        super().__init__('tidybot_nav2')

        # ---------------- ROS ----------------
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.cmd_pub  = self.create_publisher(Twist, 'cmd_vel', 10)

        self.create_subscription(Image, '/limo/depth_camera_link/image_raw', self.image_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(GoalStatusArray,
                                 '/navigate_to_pose/_action/status',
                                 self.status_cb, 10)

        self.bridge = CvBridge()

        # ---------------- PARAMETERS ----------------
        self.patch = (-1.04, 0.72)

        # ---------------- STATE ----------------
        self.stage = "INIT"
        self.goal_active = False
        self.goal_reached = False

        self.rotation_start = None
        self.target = None
        self.push_start = None

        # ---------------- SENSORS ----------------
        self.front_distance = float('inf')
        self.last_frame = None
        self.cubes = []

        self.get_logger().info("🚀 Nav2 + Behaviour controller started")
        self.timer = self.create_timer(0.1, self.loop)

    # ============================================================
    # NAV STATUS
    # ============================================================
    def status_cb(self, msg):
        if len(msg.status_list) > 0:
            status = msg.status_list[-1].status
            if status == 4:
                self.goal_reached = True

    # ============================================================
    # LASER
    # ============================================================
    def scan_cb(self, msg):
        ranges = msg.ranges
        mid = len(ranges)//2
        seg = [r for r in ranges[mid-10:mid+10] if 0.05 < r < 5.0]
        self.front_distance = min(seg) if seg else float('inf')

    # ============================================================
    # VISION
    # ============================================================
    def image_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # BROWN DETECTION
        m1 = cv2.inRange(hsv, (5,40,40), (15,180,180))
        m2 = cv2.inRange(hsv, (10,80,80), (25,255,255))
        mask = cv2.bitwise_or(m1, m2)

        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

        cubes = []
        for c in cnts:
            if cv2.contourArea(c) < 100:
                continue
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cubes.append((cx,cy))

        self.last_frame = frame
        self.cubes = cubes

    # ============================================================
    # NAV HELPERS
    # ============================================================
    def send_initial_pose(self):
        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.pose.position.x = 0.0
        pose.pose.pose.position.y = 0.0
        pose.pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)
        print("📍 Initial pose set")

    def send_goal(self, xy):
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = xy[0]
        goal.pose.position.y = xy[1]
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)
        print("🎯 Goal sent")

    # ============================================================
    # MAIN LOOP
    # ============================================================
    def loop(self):

        twist = Twist()

        # ---------------- INIT ----------------
        if self.stage == "INIT":
            self.send_initial_pose()
            self.init_time = time.time()
            self.stage = "WAIT_AMCL"
            return

        if self.stage == "WAIT_AMCL":
            if time.time() - self.init_time > 3:
                self.stage = "NAV_TO_PATCH"
            return

        # ---------------- NAVIGATION ----------------
        if self.stage == "NAV_TO_PATCH":

            if not self.goal_active:
                self.send_goal(self.patch)
                self.goal_active = True
                self.goal_reached = False
                return

            if self.goal_reached:
                print("📍 Arrived at patch")
                self.stage = "SCAN"
                self.rotation_start = time.time()
                self.goal_active = False
                return

            return

        # ---------------- SCAN ----------------
        if self.stage == "SCAN":
            twist.angular.z = 0.3
            self.cmd_pub.publish(twist)

            if len(self.cubes) > 0:
                self.target = max(self.cubes, key=lambda c: c[0])
                print("🎯 Target found")
                self.stage = "ALIGN"
                return

            if time.time() - self.rotation_start > 10:
                print("✅ No cubes left")
                self.stage = "DONE"
                return

            return

        # ---------------- ALIGN ----------------
        if self.stage == "ALIGN":

            # ❗ Always re-select target every frame
            if len(self.cubes) == 0:
                print("❌ Lost target → rescanning")
                self.stage = "SCAN"
                return

            frame_h, frame_w = self.last_frame.shape[:2]

            # pick closest-to-center cube (better than right-most)
            target = min(self.cubes, key=lambda c: abs(c[0] - frame_w//2))
            if len(self.cubes) > 0:
                frame_h, frame_w = self.last_frame.shape[:2]
                target = min(self.cubes, key=lambda c: abs(c[0] - frame_w//2))
                cx = target[0]

                error = cx - frame_w//2
                twist.angular.z = -0.002 * error

            error = cx - frame_w//2
            twist.angular.z = np.clip(-0.004 * error, -0.4, 0.4)

            # ✅ MUCH more forgiving threshold
            if abs(error) < 40:
                print("🔄 Aligned → pushing")
                self.stage = "PUSH"
                self.push_start = time.time()

            self.cmd_pub.publish(twist)
            return

        # ---------------- PUSH ----------------
        if self.stage == "PUSH":
            twist.linear.x = 0.2

            if len(self.cubes) > 0:
                frame_h, frame_w = self.last_frame.shape[:2]
                cx = self.cubes[0][0]
                error = cx - frame_w//2
                twist.angular.z = -0.002 * error

            if self.front_distance < 0.3 or \
               time.time() - self.push_start > 5:
                print("🧱 Push complete → returning")
                self.stage = "NAV_TO_PATCH"
                self.goal_active = False
                self.target = None
                return

            self.cmd_pub.publish(twist)
            return

        # ---------------- DONE ----------------
        if self.stage == "DONE":
            self.cmd_pub.publish(Twist())


# ============================================================
def main():
    rclpy.init()
    node = TidyBotNav2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()