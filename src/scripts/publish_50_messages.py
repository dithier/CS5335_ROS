#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

bridge = CvBridge()

def img_publisher(data, n):
    pub = rospy.Publisher("/webcam/image_raw", Image, queue_size=10)
    rospy.init_node("img_publisher", anonymous=True)
    rate = rospy.Rate(0.7)
    i = 0
    rate.sleep()
    while not rospy.is_shutdown():
        if i < n:
            print("publishing ", i)
            cv_img = data[i]
            img = bridge.cv2_to_imgmsg(cv_img, "bgr8")
            img.header.frame_id = "webcam"
            pub.publish(img)
            i += 1
        rate.sleep()

if __name__ == "__main__":
    try:
        path = "/home/ithier/catkin_ws/src/CS5335_ROS/src/scripts/"
        filename = path + "50_images.npz"
        with np.load(filename) as array:
            data = array["images"]
        img_publisher(data, len(data))
    except rospy.ROSInterruptException:
        pass