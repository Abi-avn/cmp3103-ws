#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import cv2
import numpy as np


class ColourChaser(Node):

    def __init__(self):
        super().__init__('colour_chaser')

        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 1)

        self.create_subscription(
            Image,
            '/limo/depth_camera_link/image_raw',
            self.camera_callback,
            1
        )

        self.br = CvBridge()

    def camera_callback(self, data):

        frame = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # BLUE MASK
        lower_blue = (100,120,70)
        upper_blue = (130,255,255)

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Remove noise
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        twist = Twist()

        if len(contours) > 0:

            largest = contours[0]

            cv2.drawContours(frame, [largest], -1, (255,255,0), 3)

            M = cv2.moments(largest)

            if M['m00'] > 0:

                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                print(f"Centroid: ({cx},{cy})")

                cv2.circle(frame,(cx,cy),8,(0,255,0),-1)

                width = data.width

                if cx < width/3:
                    twist.angular.z = 0.4

                elif cx > 2*width/3:
                    twist.angular.z = -0.4

                else:
                    twist.angular.z = 0.0
                    twist.linear.x = 0.1

        else:
            print("No blue object detected")
            twist.angular.z = 0.3

        self.pub_cmd_vel.publish(twist)

        frame_small = cv2.resize(frame,(0,0),fx=0.4,fy=0.4)
        mask_small = cv2.resize(mask,(0,0),fx=0.4,fy=0.4)

        cv2.imshow("Camera",frame_small)
        cv2.imshow("Blue Mask",mask_small)
        cv2.waitKey(1)


def main(args=None):

    print("Starting colour_chaser")

    rclpy.init(args=args)

    node = ColourChaser()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()