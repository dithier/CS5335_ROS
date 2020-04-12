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
        self.run = True
        self.img_num = 0
        self.path = "/home/ithier/catkin_ws/src/CS5335_ROS/images/calibration/"

    def image_callback(self, msg):
        if self.run:
            try:
                # Convert your ROS Image message to OpenCV2
                cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
                cv2.imshow("video", cv2_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.quit_windows()
                if key == ord('s'):
                    self.save_frame(cv2_img)
            except CvBridgeError as e:
                print(e)
    def quit_windows(self):
        print("Quitting")
        cv2.destroyAllWindows()
        self.run = False
    def save_frame(self, frame):
        img_name = self.path + str(self.img_num) + ".jpeg"
        print("image name", img_name)
        cv2.imwrite(img_name, frame)
        self.img_num += 1
        print("Wrote image to file")

def main():
    camera_node = CameraNode()
    rospy.init_node('display_video')
    print("starting node")
    # Define your image topic
    image_topic = "/webcam/image_raw"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, camera_node.image_callback)
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()
