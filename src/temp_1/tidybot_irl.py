#!/usr/bin/env python

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import time


class TidybotIRL(Node):
    def __init__(self):
        super().__init__('tidybot_task_irl')

        self.create_subscription(
            Image,
            '/limo/depth_camera_link/image_raw',
            self.camera_callback,
            1
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.br = CvBridge()

        # Behaviour attributes
        self.push_to_edge = False
        self.extra_push = False
        self.reversing = False

        self.last_arena_seen_time = 0.0
        self.last_obj_seen_time = 0.0

    def camera_callback(self, data):

        try:
            cv_image = self.br.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError:
            return
        # Convert image to hsv from BGR
        hsv_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)


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

        twist = Twist()
        now = time.time()

    
       # Push cube until the red patch disappears 
        if self.push_to_edge:
            twist.linear.x = 0.15
            self.cmd_pub.publish(twist)

            red_pixels_bottom = np.sum(mask_red[int(h * 0.8):, :])

            if red_pixels_bottom < 50:
                # Reached edge → start extra push
                self.push_to_edge = False
                self.extra_push = True
                self.extra_push_start = now

            self._visualise(cv_image, mask_red, mask_objects)
            return

         # Push the robot a little further to push the cube out of patch 
        if self.extra_push:
            if now - self.extra_push_start < 2.0:
                twist.linear.x = 0.15
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
            else:
                # Done extra push → reverse
                self.extra_push = False
                self.reversing = True
                self.reverse_start = now

            self._visualise(cv_image, mask_red, mask_objects)
            return

        #   Reverse into the patch
        if self.reversing:
            if now - self.reverse_start < 2.5:
                twist.linear.x = -0.1
                self.cmd_pub.publish(twist)
            else:
                self.reversing = False

            self._visualise(cv_image, mask_red, mask_objects)
            return

        
        #'Search and targetting 

        target_found = False
        # Use a rectangle roi to spot object bounding box.
        if len(red_contours) > 0:
            self.last_arena_seen_time = now

            # arena is the largest contour
            arena_contour = max(red_contours, key=cv2.contourArea)
            x, y, rw, rh = cv2.boundingRect(arena_contour)

            cv2.rectangle(cv_image, (x, y), (x+rw, y+rh), (0, 0, 255), 2)

            # Extract only objects INSIDE the arena
            roi_objects = mask_objects[y:y+rh, x:x+rw]

            obj_contours, _ = cv2.findContours(
                roi_objects.copy(),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # If the detected object has a big enough area it is detected
            if len(obj_contours) > 0:
                oc = max(obj_contours, key=cv2.contourArea)
                area = cv2.contourArea(oc)

                if area > 120:
                    M = cv2.moments(oc)
                    if M["m00"] != 0:
                        ox = int(M["m10"] / M["m00"]) + x
                        oy = int(M["m01"] / M["m00"]) + y

                        cv2.circle(cv_image, (ox, oy), 10, (255, 0, 0), -1)

                        self.last_obj_seen_time = now
                        target_found = True

                        center_x = w // 2
                        error = ox - center_x

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
                self.push_to_edge_start = now
            else:
                # Rotate if not
                twist.angular.z = 0.3
                self.cmd_pub.publish(twist)

        self._visualise(cv_image, mask_red, mask_objects)

    # Display for debugging
    def _visualise(self, cv_image, mask_red, mask_objects):
        cv2.imshow("Camera", cv2.resize(cv_image, (0, 0), fx=0.4, fy=0.4))
        cv2.imshow("Red Arena Mask", mask_red)
        cv2.imshow("Object Mask", mask_objects)
        cv2.waitKey(1)


def main(args=None):
    print("Starting tidybot irl")
    cv2.startWindowThread()

    rclpy.init(args=args)
    node = TidybotIRL()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()