#!/usr/bin/env python3

import rclpy, time, cv2, numpy as np
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge


class TidybotComplexity2(Node):

    def __init__(self):
        super().__init__('tidybot_task_2')

        # ---------------- ROS ----------------
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        self.create_subscription(Image, '/limo/depth_camera_link/image_raw', self.image_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        self.bridge = CvBridge()

        # ---------------- PARAMETERS ----------------
        self.patch = (-1.04, 0.72)

        # ---------------- STATE ----------------
        self.state = "INIT"
        self.nav_done = False

        # ---------------- DETECTION MEMORY ----------------
        self.last_blue_seen_time = 0.0   # ✅ renamed (was brown logic conceptually)
        self.last_arena_seen_time = 0.0
        self.last_red_seen_time = 0.0

        # ---------------- PUSH LOGIC ----------------
        self.pushing_forward = False
        self.reversing = False
        self.push_start = 0.0

        # ---------------- STOP LOGIC ----------------
        self.no_cube_start = None
        self.STOP_TIME = 6.0

        # ---------------- FINAL SCAN ----------------
        self.final_scan_start = None
        self.FINAL_SCAN_TIME = 6.0

        self.front_distance = float('inf')

        self.get_logger().info("🚀 HYBRID TIDYBOT STARTED")

    # ============================================================
    # NAVIGATION
    # ============================================================

    def send_initial_pose(self):
        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)

    def send_goal(self):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()

        goal.pose.pose.position.x = self.patch[0]
        goal.pose.pose.position.y = self.patch[1]
        goal.pose.pose.orientation.w = 1.0

        self.nav_client.wait_for_server()
        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response)

        self.nav_start = time.time()
        print("🎯 Going to patch")

    def goal_response(self, future):
        handle = future.result()

        if not handle.accepted:
            print("❌ Goal rejected")
            self.state = "SEARCH"
            return

        result_future = handle.get_result_async()
        result_future.add_done_callback(self.nav_done_cb)

    def nav_done_cb(self, future):
        print("📍 At patch")
        self.nav_done = True

    # ============================================================
    # LIDAR
    # ============================================================

    def scan_callback(self, msg):
        ranges = msg.ranges
        mid = len(ranges)//2
        seg = [r for r in ranges[mid-10:mid+10] if 0.05 < r < 5.0]
        self.front_distance = min(seg) if seg else float('inf')

    # ============================================================
    # CAMERA + FSM
    # ============================================================

    def image_callback(self, data):

        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]

        # ---------------- RED PATCH ----------------
        mask_red = cv2.inRange(hsv, (0,120,70), (10,255,255)) | \
                   cv2.inRange(hsv, (170,120,70), (180,255,255))

        mask_red[0:int(h*0.3), :] = 0

        kernel = np.ones((5,5), np.uint8)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

        red_contours,_ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        now = time.time()

        if len(red_contours) > 0:
            self.last_red_seen_time = now

        # ---------------- BLUE (TASK 2 TARGET) ----------------
        mask_blue = cv2.inRange(hsv, (90,120,70), (130,255,255))

        twist = Twist()

        # ============================================================
        # INIT
        # ============================================================
        if self.state == "INIT":
            self.send_initial_pose()
            time.sleep(3)
            self.send_goal()
            self.state = "WAIT_NAV"
            return

        # ============================================================
        # WAIT NAV
        # ============================================================
        if self.state == "WAIT_NAV":
            if self.nav_done or (time.time() - self.nav_start > 10):
                self.state = "SEARCH"
            return

        # ============================================================
        # FINAL SCAN
        # ============================================================
        if self.state == "FINAL_SCAN":

            twist.angular.z = 0.5
            self.cmd_pub.publish(twist)

            if len(red_contours) > 0:
                arena = max(red_contours, key=cv2.contourArea)
                x,y,rw,rh = cv2.boundingRect(arena)
                roi = mask_blue[y:y+rh, x:x+rw]   # ✅ BLUE used here

                blue_cnts,_ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if blue_cnts:
                    print("❗ Cube found during final scan → resume")
                    self.state = "SEARCH"
                    self.no_cube_start = None
                    return

            if time.time() - self.final_scan_start > self.FINAL_SCAN_TIME:
                print("🏁 CONFIRMED COMPLETE")
                self.state = "DONE"

            return

        # ============================================================
        # PUSH FORWARD (BLUE VERSION TASK 2)
        # ============================================================
        if self.pushing_forward:

            RED_LOST_TIMEOUT = 1.0

            if now - self.last_red_seen_time < RED_LOST_TIMEOUT:
                twist.linear.x = 0.2
                self.cmd_pub.publish(twist)
            else:
                self.get_logger().info("Blue lost → stop pushing")
                self.pushing_forward = False
                self.reversing = True
                self.reverse_start = now
            return

        # ============================================================
        # REVERSE
        # ============================================================
        if self.reversing:
            if now - self.reverse_start < 1.5:
                twist.linear.x = -0.1
                self.cmd_pub.publish(twist)
            else:
                self.reversing = False
                self.send_goal()
                self.state = "WAIT_NAV"
            return

        # ============================================================
        # SEARCH / TARGET
        # ============================================================

        target_found = False

        if len(red_contours) > 0:
            self.last_arena_seen_time = now

            arena = max(red_contours, key=cv2.contourArea)
            x,y,rw,rh = cv2.boundingRect(arena)

            roi = mask_blue[y:y+rh, x:x+rw]   # ✅ BLUE used here

            blue_cnts,_ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if blue_cnts:
                bc = max(blue_cnts, key=cv2.contourArea)

                if cv2.contourArea(bc) > 100:
                    M = cv2.moments(bc)
                    if M["m00"] != 0:

                        bx = int(M["m10"]/M["m00"]) + x

                        self.last_blue_seen_time = now
                        target_found = True

                        error = bx - w//2

                        twist.angular.z = -error / 300.0
                        twist.linear.x = 0.15

                        self.cmd_pub.publish(twist)

        # ============================================================
        # PUSH TRIGGER
        # ============================================================
        if not target_found:
            if now - self.last_blue_seen_time < 0.2:
                self.pushing_forward = True
                self.push_start = now
            else:
                twist.angular.z = 0.3
                self.cmd_pub.publish(twist)

        # ============================================================
        # STOP → FINAL SCAN
        # ============================================================
        if len(red_contours) > 0 and not target_found:

            if self.no_cube_start is None:
                self.no_cube_start = now

            elif now - self.no_cube_start > self.STOP_TIME:
                print("🔍 Starting final scan...")
                self.state = "FINAL_SCAN"
                self.final_scan_start = time.time()

        else:
            self.no_cube_start = None

        # ============================================================
        # DONE
        # ============================================================
        if self.state == "DONE":
            self.cmd_pub.publish(Twist())
            exit()

        # ============================================================
        # SAFETY
        # ============================================================
        if self.front_distance < 0.25:
            twist.linear.x = 0.0
            twist.angular.z = 0.5
            self.cmd_pub.publish(twist)

        cv2.putText(frame, self.state, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("view", cv2.resize(frame,(0,0),fx=0.4,fy=0.4))
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = TidybotComplexity2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()