import math
import time
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from nav2_simple_commander.robot_navigator import BasicNavigator
from tf_transformations import quaternion_from_euler

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        self.bridge = CvBridge()

        self.create_subscription(
            Image,
            "/limo/depth_camera_link/image_raw",
            self.camera_callback,
            10
        )

        self.large_cube_center = None
        self.small_cube_center = None
        self.large_left_x = None

        self.last_seen_large = 0
        self.last_seen_small = 0

        self.frame_width = None

    def cubes_visible(self):
        now = time.time()
        return (now - self.last_seen_large < 1.2 and
                now - self.last_seen_small < 1.2)

    def camera_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if self.frame_width is None:
            self.frame_width = frame.shape[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        mask_large = cv2.inRange(hsv, (5, 48, 40), (20, 232, 255))
        mask_small = cv2.inRange(hsv, (5, 80, 40), (20, 255, 104))

        mask_large = cv2.morphologyEx(mask_large, cv2.MORPH_OPEN, kernel)
        mask_large = cv2.morphologyEx(mask_large, cv2.MORPH_CLOSE, kernel)

        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, kernel)
        mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel)

        self.large_cube_center = None
        self.small_cube_center = None
        self.large_left_x = None

        # ---------------- LARGE CUBE ----------------
        contours, _ = cv2.findContours(mask_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)

            if area <= 2500:
                continue

            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

            leftmost = tuple(c[c[:, :, 0].argmin()][0])
            self.large_left_x = leftmost[0]

            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                self.large_cube_center = (cx, cy)
                self.last_seen_large = time.time()

                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            cv2.circle(frame, leftmost, 6, (0, 255, 255), -1)

        # ---------------- SMALL CUBE ----------------
        contours, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)

            if not (100 < area < 4000):
                continue

            cv2.drawContours(frame, [c], -1, (255, 128, 0), 2)

            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                self.small_cube_center = (cx, cy)
                self.last_seen_small = time.time()

                cv2.circle(frame, (cx, cy), 5, (255, 128, 0), -1)

        # ---------------- DEBUG ----------------
        now = time.time()

        cv2.putText(frame,
                    f"L:{now-self.last_seen_large:.2f}s S:{now-self.last_seen_small:.2f}s",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)

        cv2.imshow("DEBUG VIEW", frame)
        cv2.imshow("MASK LARGE", mask_large)
        cv2.imshow("MASK SMALL", mask_small)
        cv2.waitKey(1)


def pose_from_xytheta(stamp, x, y, theta):
    q = quaternion_from_euler(0, 0, theta)
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.header.stamp = stamp
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation.z = q[2]
    pose.pose.orientation.w = q[3]
    return pose


waypoints_data = [
    [1.0, 0.0, math.pi/2],
    [1.3, 1.0, 3*math.pi/4],
    [0.8, 1.45, -3*math.pi/4],
]


def main():
    rclpy.init()

    node = CameraNode()
    cmd_vel_pub = node.create_publisher(Twist, "/cmd_vel", 10)

    navigator = BasicNavigator()

    stamp = navigator.get_clock().now().to_msg()
    navigator.setInitialPose(pose_from_xytheta(stamp, 0.0, 0.0, 0.0))
    navigator.waitUntilNav2Active()

    waypoints = []
    for x, y, theta in waypoints_data:
        stamp = navigator.get_clock().now().to_msg()
        waypoints.append(pose_from_xytheta(stamp, x, y, theta))

    navigator.followWaypoints(waypoints)

    state = "NAVIGATING"
    back_start = None

    while rclpy.ok():

        rclpy.spin_once(node, timeout_sec=0.01)

        # ---------------- NAVIGATION ----------------
        if state == "NAVIGATING" and navigator.isTaskComplete():
            print("Reached waypoint 3 → backing up")
            state = "BACK_UP"
            back_start = time.time()

        # ---------------- BACK UP ----------------
        elif state == "BACK_UP":
            twist = Twist()
            twist.linear.x = -0.1
            cmd_vel_pub.publish(twist)

            if time.time() - back_start > 2.0:
                state = "SEARCH"

        # ---------------- SEARCH ----------------
        elif state == "SEARCH":
            twist = Twist()

            if node.cubes_visible():
                state = "ALIGN"
                continue

            twist.angular.z = -0.2
            cmd_vel_pub.publish(twist)

        # ---------------- ALIGN ----------------
        elif state == "ALIGN":
            twist = Twist()

            if node.cubes_visible() and node.small_cube_center is not None:

                frame_center = node.frame_width // 2
                small_x, _ = node.small_cube_center

                error = small_x - frame_center

                if abs(error) > 20:
                    twist.angular.z = -0.002 * error
                    cmd_vel_pub.publish(twist)
                else:
                    print("Aligned → CURVED PUSH")
                    state = "PUSH_LEFT_EDGE"

            else:
                state = "SEARCH"

        # ---------------- CURVED PUSH (UPDATED) ----------------
        elif state == "PUSH_LEFT_EDGE":
            twist = Twist()

            if node.small_cube_center is None:
                cmd_vel_pub.publish(Twist())
                continue

            small_x, _ = node.small_cube_center
            frame_center = node.frame_width // 2
            error = small_x - frame_center

            # 🔥 ARC MOTION (like your friend's robot)
            twist.linear.x = 0.10
            twist.angular.z = 0.18
            twist.angular.z += (-0.0015 * error)

            cmd_vel_pub.publish(twist)

            # stop condition (cube pushed far left in view)
            if small_x < frame_center * 0.35:
                print("Done pushing in arc → DONE")
                state = "DONE"

        # ---------------- DONE ----------------
        elif state == "DONE":
            cmd_vel_pub.publish(Twist())

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()