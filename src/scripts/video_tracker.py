import sys
import signal
import math
import yaml
import rospy
import cv2
import numpy as np
from std_msgs.msg import Float64
from cv_bridge import CvBridge
from ar_track_alvar_msgs.msg import AlvarMarkers, AlvarMarker
from geometry_msgs.msg import Pose 
from sensor_msgs.msg import Image
from tracking_class import Tracking
from PIL import Image as im

class Video_Pose_Publisher:
    # Function to initialize
    def __init__(self):
        self.i=0
        self.ar_pose = Pose()
        self.height = 450
        self.count = 0
	self.images = []
        self.Tag_Pose_Publisher = rospy.Publisher('/Ar_Pose', Pose, queue_size=1)
        
        

# callback for the pose received from the ar tag
    def ar_tag_callback(self, msg):
        print("ar:",self.i)
        self.i = self.i+1
        markers = msg.markers
        if len(markers) > 0:
            msg = markers[0]
            self.ar_pose = msg.pose.pose
        else:
            self.ar_pose = Pose()
        self.Tag_Pose_Publisher.publish(self.ar_pose)


    # callback for the pose received from the object tracker 
    def tracking_callback(self, msg):
        #print("our tracking")
        bridge = CvBridge()
	cv_img = bridge.imgmsg_to_cv2(msg,desired_encoding="passthrough")
        #img=im.fromarray(cv_img)
        print("count:",self.count)
        self.count = self.count+1
	h, w = cv_img.shape[:2]
	r = self.height / float(h)
	dim = (int(w * r), self.height)
	img = cv2.resize(cv_img, dim, interpolation=cv2.INTER_AREA)
        self.images.append(img)
        
        

def main():
    rospy.init_node("videoTracker")
    video_Pose_Publisher = Video_Pose_Publisher()
    ###
    path = "./"
    filename = path + "webcam.yaml"
    with open(filename, "r") as file_handle:
       	calib_data = yaml.load(file_handle)
    cameramtx = calib_data["camera_matrix"]["data"]
    cameramtx = np.array(cameramtx).reshape((3,3))

    # load image that has target in it at 0 degree rotation
    bw_target = cv2.imread("../../images/2.jpg", 0)
    h, w = bw_target.shape[:2]
    r = video_Pose_Publisher.height / float(h)
    dim = (int(w * r), video_Pose_Publisher.height)
    bw_target = cv2.resize(bw_target, dim, interpolation=cv2.INTER_AREA)

    # get desired contour for perfect match
    desired = cv2.imread("../../images/desired_cnt.png")
    desired_cnt = Tracking.get_desired_cnt(desired)
    ###
    ##the subscribers
    rospy.Subscriber('/ar_pose_marker', AlvarMarkers, video_Pose_Publisher.ar_tag_callback)
    rospy.Subscriber('/webcam/image_raw', Image, video_Pose_Publisher.tracking_callback)
    Tracking_Publisher = rospy.Publisher('/Tracking_Pose', Pose, queue_size=1)
    rate = rospy.Rate(30)
    i=0
    while not rospy.is_shutdown():
        
	if video_Pose_Publisher.images:
            # process the frame
            tracking_obj = Tracking(bw_target, desired_cnt, cameramtx)

            # show results
            _, quaternion, position = tracking_obj.process_frame(video_Pose_Publisher.images.pop(0))
            print("track:",i)
            i = i+1
            image_pose = Pose()
            image_pose.orientation.x = quaternion[0]
            image_pose.orientation.y = quaternion[1]
            image_pose.orientation.z = quaternion[2]
            image_pose.orientation.w = quaternion[3]

            image_pose.position.x = position[0]
            image_pose.position.y = position[1]
            image_pose.position.z = position[2]
            
            Tracking_Publisher.publish(image_pose)  
           
        rate.sleep()






if __name__=='__main__':
    main()
