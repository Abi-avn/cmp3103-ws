#!/usr/bin/env python3
"""
Detector‑3D Enhanced TidyBot
----------------------------
Combines an RGB‑D color‑based detector with Nav 2 navigation.
Publishes 3‑D cube and patch positions in the map frame so the
robot can align and push accurately.

You can run this on its own (only shows detections), or keep
Nav 2 running and subscribe to /object_location in your tidybot
controller.
"""

import rclpy, math, cv2, numpy as np
from rclpy.node import Node
from rclpy import qos
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header

import image_geometry


class Detector3D(Node):
    """
    Detects red or brown color blobs in RGB, projects them into
    3‑D camera coordinates using the depth image, transforms to
    map/odom coordinates, and publishes PoseStamped results.
    """

    def __init__(self, real_robot=False, detect_color="brown"):
        super().__init__('detector3d')

        self.bridge = CvBridge()
        self.real_robot = real_robot
        self.detect_color = detect_color     # "brown" or "red"
        self.min_area = 120                  # ignore tiny blobs
        self.global_frame = 'map'
        self.camera_frame = 'depth_link'

        # camera models and aspect ratio
        self.ccam = None
        self.dcam = None
        self.color2depth_aspect = None
        self.image_depth_ros = None

        # topic remaps for real / sim
        if real_robot:
            cinfo = '/camera/color/camera_info'
            dinfo = '/camera/depth/camera_info'
            cimg  = '/camera/color/image_raw'
            dimg  = '/camera/depth/image_raw'
            self.camera_frame = 'camera_color_optical_frame'
        else:
            cinfo = '/limo/depth_camera_link/camera_info'
            dinfo = '/limo/depth_camera_link/depth/camera_info'
            cimg  = '/limo/depth_camera_link/image_raw'
            dimg  = '/limo/depth_camera_link/depth/image_raw'

        # --- Subscribers ---
        self.create_subscription(CameraInfo, cinfo, self.ccam_cb, qos_profile=qos.qos_profile_sensor_data)
        self.create_subscription(CameraInfo, dinfo, self.dcam_cb, qos_profile=qos.qos_profile_sensor_data)
        self.create_subscription(Image, dimg, self.depth_cb, qos_profile=qos.qos_profile_sensor_data)
        self.create_subscription(Image, cimg, self.color_cb, qos_profile=qos.qos_profile_sensor_data)

        # --- Publisher ---
        self.obj_pub = self.create_publisher(PoseStamped, '/object_location', 10)

        # --- TF ---
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)

        self.get_logger().info("🚀 3‑D Detector started – publishing /object_location")

    # ============================================================
    # Camera callbacks
    # ============================================================
    def ccam_cb(self, data):
        if self.ccam is None:
            self.ccam = image_geometry.PinholeCameraModel()
            self.ccam.fromCameraInfo(data)
            self.calc_aspect()

    def dcam_cb(self, data):
        if self.dcam is None:
            self.dcam = image_geometry.PinholeCameraModel()
            self.dcam.fromCameraInfo(data)
            self.calc_aspect()

    def calc_aspect(self):
        # alignment ratio between color & depth imagery
        if self.ccam and self.dcam and self.color2depth_aspect is None:
            self.color2depth_aspect = (
                math.atan2(self.ccam.width, 2*self.ccam.fx())/self.ccam.width
            ) / (
                math.atan2(self.dcam.width, 2*self.dcam.fx())/self.dcam.width
            )

    # ============================================================
    # Image callbacks
    # ============================================================
    def depth_cb(self, data):
        self.image_depth_ros = data

    def color_cb(self, data):
        if not (self.ccam and self.dcam and self.image_depth_ros):
            return  # wait for calibration

        # --- Convert to OpenCV ---
        color = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(self.image_depth_ros, '32FC1')
        if self.real_robot:
            depth /= 1000.0  # mm→m on real Limo

        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

        # --- color mask selection ---
        if self.detect_color == "brown":
            mask = cv2.inRange(hsv, (5,50,50), (25,255,255))
        elif self.detect_color == "red":
            mask = cv2.inRange(hsv,(0,70,50),(10,255,255)) + cv2.inRange(hsv,(170,70,50),(180,255,255))
        else:
            return

        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i,c in enumerate(cnts):
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            M = cv2.moments(c)
            if M['m00']==0: continue

            cy = int(M['m01']/M['m00'])
            cx = int(M['m10']/M['m00'])
            # convert 2‑D pixel → 3‑D world
            pose_local = self.pixel_to_world((cx,cy), color, depth)
            if pose_local is None:
                continue

            # Transform to map/odom frame
            try:
                t = self.tf_buf.lookup_transform(self.global_frame,
                                                 self.camera_frame,
                                                 rclpy.time.Time())
                pose_global = do_transform_pose(pose_local, t)
            except Exception as e:
                self.get_logger().warn(f"TF lookup failed {e}")
                continue

            msg = PoseStamped()
            msg.header.frame_id = self.global_frame
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose = pose_global
            self.obj_pub.publish(msg)

            # Debug print
            p = pose_global.position
            print(f"🟫 object #{i}: map coords ≈ ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})")

            cv2.circle(color, (cx,cy),5,(0,255,0),2)

        # --- Visualisation windows ---
        cv2.imshow("RGB Image", cv2.resize(color,(0,0),fx=0.5,fy=0.5))
        depth_v = (depth*0.1).clip(0,1)
        cv2.imshow("Depth", depth_v)
        cv2.waitKey(1)

    # ============================================================
    # Core utility – 2D pixel → Pose in camera frame + depth
    # ============================================================
    def pixel_to_world(self, pix, img, depth_img):
        try:
            u,v = pix
            # bound check
            if v>=depth_img.shape[0] or u>=depth_img.shape[1]:
                return None
            z = float(depth_img[int(v)][int(u)])
            if z==0 or math.isnan(z):
                return None

            ray = np.array(self.ccam.projectPixelTo3dRay((u,v)))
            ray /= ray[2]  # normalize s.t. z = 1
            X,Y,Z = ray*z
            pose = Pose()
            pose.position = Point(x=X,y=Y,z=Z)
            pose.orientation = Quaternion(w=1.0)
            return pose
        except Exception as e:
            self.get_logger().warn(f"Projection fail {e}")
            return None


def main(args=None):
    rclpy.init(args=args)
    node = Detector3D(real_robot=False, detect_color="brown")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()