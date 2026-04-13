#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
import time
import sys


class TidybotComplexity1(Node):

    def __init__(self):
        super().__init__('tidybot_task_1')

        
        self.create_subscription(
            Image,
            '/limo/depth_camera_link/image_raw',
            self.camera_callback,
            1
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        self.br = CvBridge()

        # Navigation
        self.patch = (-1.04, 0.72)
        self.state = "INIT"
        self.nav_done = False
        self.init_start_time = None

        # Behaviour attributes
        self.push_to_edge = False
        self.extra_push = False
        self.reversing = False

        self.last_obj_seen_time = 0.0
        self.last_arena_seen_time = 0.0

        # Exit condition
        self.victory_start = None
        self.VICTORY_TIME = 60.0

        self.get_logger().info(" TidybotComplexity1 STARTED")

   
    # publish inital robot pose
    def send_initial_pose(self):
        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)
        print("Initial pose set")
   # send goal to nav2 
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
        print("Navigating to patch")
    # handle response 
    def goal_response(self, future):
        handle = future.result()

        if not handle.accepted:
            print("Goal rejected")
            self.state = "SEARCH"
            return

        result_future = handle.get_result_async()
        result_future.add_done_callback(self.nav_done_cb)
    # check if robot has arrived at the patch
    def nav_done_cb(self, future):
        print("Arrived at patch")
        self.nav_done = True


    def camera_callback(self, data):

        try:
            cv_image = self.br.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError:
            return
        # Convert image to hsv from BGR
        hsv_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        #set the inital pose to initialte /amcl
        if self.state == "INIT":

            if self.init_start_time is None:
                self.send_initial_pose()
                self.init_start_time = time.time()
                return

            if time.time() - self.init_start_time > 3.0:
                self.send_goal()
                self.state = "WAIT_NAV"

            return
        # Wait for the nav2 to move close to the patch
        if self.state == "WAIT_NAV":
            if self.nav_done or (time.time() - self.nav_start > 10):
                print(" Switching to SEARCH")
                self.state = "SEARCH"
            return

        # Detection

        # Red masks

        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)

        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)

        mask_red = mask1 | mask2

        h, w = mask_red.shape

        # Filter out the top 30% to ignore the red wall patch
        mask_red[0:int(h * 0.3), :] = 0

        # Noise removal 
        kernel = np.ones((5, 5), np.uint8)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

        red_contours, _ = cv2.findContours(
            mask_red.copy(),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        # Remove floor mask
        floor_mask = cv2.inRange(
            hsv_img,
            np.array([0, 0, 0]),
            np.array([180, 80, 200])
        )
        # Object mask is everything that isn't red or the floor
        mask_objects = cv2.bitwise_not(mask_red)
        mask_objects = cv2.bitwise_and(mask_objects, cv2.bitwise_not(floor_mask))
        mask_objects = cv2.morphologyEx(mask_objects, cv2.MORPH_OPEN, kernel)

        now = time.time()
        twist = Twist()

        # Exit state condition 
        if self.state == "CHECK_VICTORY":
            # rotate anti-clockwise
            twist.angular.z = 0.5
            self.cmd_pub.publish(twist)

            if len(red_contours) > 0:
                arena = max(red_contours, key=cv2.contourArea)
                x, y, rw, rh = cv2.boundingRect(arena)

                roi = mask_objects[y:y+rh, x:x+rw]

                obj_contours, _ = cv2.findContours(
                    roi.copy(),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                if obj_contours:
                    oc = max(obj_contours, key=cv2.contourArea)

                    if cv2.contourArea(oc) > 150:
                        print("CUBE FOUND: Resume Search")
                        self.state = "SEARCH"
                        return

            if time.time() - self.victory_start > self.VICTORY_TIME:
                print("TASK COMPLETE: no cubes left")
                rclpy.shutdown()
                sys.exit(0)

            return

        # Push cube until the red patch disappears 
        if self.push_to_edge:
            twist.linear.x = 0.15
            self.cmd_pub.publish(twist)

            red_pixels_bottom = np.sum(mask_red[int(h * 0.8):, :])

            if red_pixels_bottom < 50:
                self.push_to_edge = False
                self.extra_push = True
                self.extra_push_start = now

            self._visualise(cv_image, mask_red, mask_objects)
            return
        # Push the robot a little further to push the cube out of patch 
        if self.extra_push:
            if now - self.extra_push_start < 2.0:
                twist.linear.x = 0.15
                self.cmd_pub.publish(twist)
            else:
                self.extra_push = False
                self.reversing = True
                self.reverse_start = now

            self._visualise(cv_image, mask_red, mask_objects)
            return
        # Reverse into the patch
        if self.reversing:
            if now - self.reverse_start < 2.5:
                twist.linear.x = -0.1
                self.cmd_pub.publish(twist)
            else:
                self.reversing = False

            self._visualise(cv_image, mask_red, mask_objects)
            return

    
        # Search and targetting
        target_found = False

        # Use a rectangle roi to spot object bounding box.
        if len(red_contours) > 0:
            self.last_arena_seen_time = now
            # arena is the largest red contour
            arena = max(red_contours, key=cv2.contourArea)
            x, y, rw, rh = cv2.boundingRect(arena)

            cv2.rectangle(cv_image, (x, y), (x+rw, y+rh), (0, 0, 255), 2)
            #   Region of interest inside the arena
            roi = mask_objects[y:y+rh, x:x+rw]

            obj_contours, _ = cv2.findContours(
                roi.copy(),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            # If the detected object has a big enough area it is detected
            if obj_contours:
                oc = max(obj_contours, key=cv2.contourArea)
                area = cv2.contourArea(oc)
                
                if area > 130:
                    M = cv2.moments(oc)

                    if M["m00"] != 0:
                        ox = int(M["m10"] / M["m00"]) + x
                        oy = int(M["m01"] / M["m00"]) + y

                        cv2.circle(cv_image, (ox, oy), 10, (0, 255, 0), -1)

                        self.last_obj_seen_time = now
                        target_found = True

                        error = ox - w // 2

                        if abs(error) > 40:
                            twist.angular.z = -float(error) / 250.0
                        else:
                            twist.linear.x = 0.15

                        self.cmd_pub.publish(twist)

        
        # Locomotion 
        
        if not target_found:
            # Target the last seen cube
            if now - self.last_obj_seen_time < 0.2:
                self.push_to_edge = True
            else:
            # Rotate if not
                twist.angular.z = 0.3
                self.cmd_pub.publish(twist)

            # Exit condition check
            if now - self.last_obj_seen_time > 5.0:
                print("NO CUBES:  Checking victory")
                self.state = "CHECK_VICTORY"
                self.victory_start = time.time()
                return

        self._visualise(cv_image, mask_red, mask_objects)


    # Display for debugging
    def _visualise(self, cv_image, mask_red, mask_objects):
        cv2.imshow("Camera", cv_image)
        cv2.imshow("Red Mask", mask_red)
        cv2.imshow("Objects", mask_objects)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = TidybotComplexity1()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()