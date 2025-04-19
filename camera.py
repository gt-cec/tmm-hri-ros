#!/usr/bin/env python

import pyrealsense2 as rs
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    rospy.init_node('realsense_camera_node', anonymous=True)

    color_pub = rospy.Publisher("/camera/color/image_raw", Image, queue_size=1)
    depth_pub = rospy.Publisher("/camera/depth/image_rect_raw", Image, queue_size=1)

    bridge = CvBridge()

    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start the stream
    pipeline.start(config)

    rate = rospy.Rate(10)

    try:
        while not rospy.is_shutdown():
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Convert to ROS images
            color_msg = bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
            depth_msg = bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")

            # Publish images
            color_pub.publish(color_msg)
            depth_pub.publish(depth_msg)

            rate.sleep()
    finally:
        pipeline.stop()

if __name__ == '__main__':
    main()
