#! /usr/bin/python

import sys
import signal
import math
import rospy
from std_msgs.msg import Float64
from ar_track_alvar_msgs.msg import AlvarMarker
from geometry_msgs.msg import Pose 

class Rotation_Error:
    # Function to initialize
    def __init__(self):
        self.x1 = 0
        self.y1 = 0
        self.z1 = 0
        self.w1 = 0
        self.x2 = 0
        self.y2 = 0
        self.z2 = 0
        self.w2 = 0
	print("hi")
	print(self.x1)
        self.data = Float64()
        self.error_Publisher = rospy.Publisher('/error', Float64, queue_size=1)

    # callback for the pose received from the ar tag
    def ar_tag_callback(self, msg):
	print("ar")
        self.x1 = msg.pose.pose.orientation.x
        self.y1 = msg.pose.pose.orientation.y
        self.z1 = msg.pose.pose.orientation.z
        self.w1 = msg.pose.pose.orientation.w
        q = self.x1*self.x2 + self.y1*self.y2 + self.z1*self.z2 + self.w1*self.w2
	print(q)
        self.data = math.acos(2*(q**2) - 1)

    # callback for the pose received from the object tracker 
    def tracking_callback(self, msg):
        self.x2 = msg.pose.orientation.x
        self.y2 = msg.pose.orientation.y
        self.z2 = msg.pose.orientation.z
        self.w2 = msg.pose.orientation.w
        q = self.x1 * self.x2 + self.y1 * self.y2 + self.z1 * self.z2 + self.w1 * self.w2
        self.data = math.acos(2 * (q ** 2) - 1)

    # Function to publish the error
    def publish_Error(self):
        self.error_Publisher.publish(self.data)

def main():
    rospy.init_node("error")
    print("main")
    rotation_error = Rotation_Error()
    rospy.Subscriber('/ar_pose_marker',AlvarMarker,rotation_error.ar_tag_callback)
    rospy.Subscriber('/topic_from_our_method',Pose,rotation_error.tracking_callback)
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        rotation_error.publish_Error()


        rate.sleep()

if __name__=='__main__':
    main()
