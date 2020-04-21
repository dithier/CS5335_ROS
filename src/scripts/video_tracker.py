#! /usr/bin/python

import yaml
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from ar_track_alvar_msgs.msg import AlvarMarkers, AlvarMarker
from geometry_msgs.msg import Pose 
from sensor_msgs.msg import Image
from tracking_class import Tracking

bridge = CvBridge()

class Video_Pose_Publisher:
    # Function to initialize
    def __init__(self, tracking_obj):
        # whether or not the bag file has start playing
        self.start = False
        # number of messages in ros bag
        self.num_messages = 1213
        # object that does our pose estimation
        self.tracking_obj = tracking_obj
        # number of messages published for ar pose
        self.ar_count = 0
        # height of resized image
        self.height = 450
        # number of messages published for our pose
        self.our_count = 0
        # ar tag publisher
        self.ar_pose_publisher = rospy.Publisher('/Ar_Pose', Pose, queue_size = 1)
        # our pose publisher
        # self.our_pose_publisher = rospy.Publisher('/Our_Pose', Pose, queue_size = 1)
        self.images = []

    # callback for the pose received from the ar tag
    def ar_tag_callback(self, msg):
        print("ar:", self.ar_count)
        if self.start:
            if self.ar_count < self.num_messages:
                self.ar_count += 1
                markers = msg.markers
                if len(markers) > 0:
                    msg = markers[0]
                    ar_pose = msg.pose.pose
                else:
                    ar_pose = Pose()
                self.ar_pose_publisher.publish(ar_pose)
            else:
                print("AR tags all processed")
        else:
            markers = msg.markers
            if len(markers) > 0:
                msg = markers[0]
                ar_pose = msg.pose.pose
                self.ar_pose_publisher.publish(ar_pose)
                self.ar_count += 1
                self.start = True

    # callback for the pose received from the object tracker 
    def tracking_callback(self, msg):
        """
        print("our tracking: ", self.our_count)
        if self.our_count < self.num_messages:
            self.our_count = self.our_count + 1
            try:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                h, w = cv_img.shape[:2]
                r = self.height / float(h)
                dim = (int(w * r), self.height)
                img = cv2.resize(cv_img, dim, interpolation=cv2.INTER_AREA)

                # show results
                _, quaternion, position = self.tracking_obj.process_frame(img)

                image_pose = Pose()

                if len(quaternion) > 0:
                    image_pose.orientation.x = quaternion[0]
                    image_pose.orientation.y = quaternion[1]
                    image_pose.orientation.z = quaternion[2]
                    image_pose.orientation.w = quaternion[3]

                    image_pose.position.x = position[0]
                    image_pose.position.y = position[1]
                    image_pose.position.z = position[2]

                self.our_pose_publisher.publish(image_pose)

            except CvBridgeError, e:
                print(e)
                image_pose = Pose()
                self.our_pose_publisher.publish(image_pose)
        else:
            print("Our pose all processed")

        """
        try:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.images.append(cv_img)

            if len(self.images) == self.num_messages:
                path = "/home/ithier/catkin_ws/src/CS5335_ROS/src/scripts/"
                filename = path + "all_images"
                np.savez(filename, images=self.images)
                print("saved images")
        except CvBridgeError, e:
            print(e)


def main():
    # get camera matrix from calibration file
    path = "/home/ithier/catkin_ws/src/CS5335_ROS/src/scripts/"
    filename = path + "webcam.yaml"

    with open(filename, "r") as file_handle:
        calib_data = yaml.load(file_handle)

    cameramtx = calib_data["camera_matrix"]["data"]
    cameramtx = np.array(cameramtx).reshape((3, 3))

    # get desired contour for perfect match
    images_path = "/home/ithier/catkin_ws/src/CS5335_ROS/images/"
    file = images_path + "desired_cnt.png"
    desired = cv2.imread(file)
    desired_cnt = Tracking.get_desired_cnt(desired)

    # load image that has target in it at 0 degree rotation
    height = 450
    file = images_path + "2.jpg"
    bw_target = cv2.imread(file, 0)
    h, w = bw_target.shape[:2]
    r = height / float(h)
    dim = (int(w * r), height)
    bw_target = cv2.resize(bw_target, dim, interpolation=cv2.INTER_AREA)

    # create tracking object
    tracking_obj = Tracking(bw_target, desired_cnt, cameramtx)

    rospy.init_node("videoTracker")

    video_Pose_Publisher = Video_Pose_Publisher(tracking_obj)

    # the subscribers
    rospy.Subscriber('/ar_pose_marker', AlvarMarkers, video_Pose_Publisher.ar_tag_callback)
    rospy.Subscriber('/webcam/image_raw', Image, video_Pose_Publisher.tracking_callback)

    rospy.spin()
        

if __name__=='__main__':
    main()
