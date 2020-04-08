#! /usr/bin/python
# Copyright (c) 2015, Rethink Robotics, Inc.

# Using this CvBridge Tutorial for converting
# ROS images to OpenCV2 images
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

# Using this OpenCV2 tutorial for saving Images:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html

# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

# Instantiate CvBridge
bridge = CvBridge()

class CameraNode():
    def __init__(self):
        self.count = 0
        self.img_num = 0
        self.path = "../images/calibration/"

    def image_callback(self, msg):
        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError, e:
            print(e)
        else:
            # only save every 10th frame
            if self.count % 10 == 0:
                # Save your OpenCV2 image as a jpeg
                img_name = self.path + str(self.img_num) + ".jpeg"
                cv2.imwrite(img_name, cv2_img)
                self.img_num += 1
                print("Wrote image to file")
            self.count += 1

def main():
    camera_node = CameraNode()
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "/webcam/image_raw"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, camera_node.image_callback)
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()