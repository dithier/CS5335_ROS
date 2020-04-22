from __future__ import division
from pose_class import HomographyandPose
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import math

# camera field of view
CAMERA_FOV = 55

# actual width (in inches) of target
TARGET_WIDTH = 9.25

# actual height (in inches) of target
TARGET_HEIGHT = 11.25

# the maximum allowed distance for matches
MATCH_DIST = 27

# the minimum number of keypoint matches there has to be to identify the target
MATCH_NUM_THRESHOLD = 6

# the number of keypoints to be used to find the best contour
KEYPOINT_NUM = 25

# threshold for matching shapes to confirm box match
SHAPE_MATCH_THRESHOLD = 0.3

# min distance that keypoints have to be apart
RADIUS = 50

# size for one unit of the axis in the reference frame of 1.jpg
AXIS_SIZE = 100

#width of the original image from the webcam before resizing
original_image_w = 1726

#height of the original image from the webcam before resizing
original_image_h = 2120

# x position of the center of the tag
ar_tag_w = 936 #center of ar tag on image 2.jpg

# y position of the center of the tag
ar_tag_h = 647 #center of ar tag on image 2.jpg

class Tracking:
    def __init__(self, bw_target, desired_cnt, cameramtx):
        self.bw_target = bw_target
        self.desired_cnt = desired_cnt
        self.cameramtx = cameramtx

    """
    process_frame
    Inputs: frame - the frame to be processed/analyzed
    Outputs: new_frame - the original frame overlayed with information. The information
    is the best fit rectangle, the box's orientation angle, offset angle, and distance from
    the target to the camera. It also has a line going from the center of the image to the center
    of the target.
    Description: This function is the bulk of the program. It is responsible for all of the logic
    for analyzing one frame and can be called repeatedly on different frames in a video to analyze
    a whole video. It does a bunch of calculations to find offset angle (the angle of the target from
    the center of the image), distance from the camera to the target, orientation of the target relative
    to the sample target image (target.jpg), a bounding box for the target, and the center of the target.
    It then draws this information on the frame and returns that new frame.
    """

    def process_frame(self, frame):
        quaternion = []
        position = []

        # get contours
        contours, roi = self.get_contours(frame)
        # get rotation angle of target
        orig_pts, target_pts = self.get_keypoints(frame, self.bw_target, roi)

        if len(orig_pts) > 0:
            bfr = self.best_contour(contours, orig_pts, self.desired_cnt)
        else:
            bfr = None

        if np.any(bfr):
            box, H = self.get_orientation(bfr, orig_pts, target_pts, self.bw_target)

            if box is not None:
                T, R, t = HomographyandPose.get_rotation_matrix(H, self.cameramtx)

                # r = Rotation.from_matrix(R)
                r = Rotation.from_dcm(R)
                quaternion = r.as_quat()
                position = self.find_position(self.bw_target,frame,H,box)

                axis = np.float32([[AXIS_SIZE, 0, 0], [0, AXIS_SIZE, 0], [0, 0, -AXIS_SIZE]]).flatten().reshape((3, 3))
                projected_pts = HomographyandPose.project_pts(T, self.cameramtx, axis)

                frame = self.draw_on_image(frame, box, projected_pts)
            else:
                print("Homography could not be found")
        else:
            print("no valid contours")
        return frame, quaternion, position

    ## Function to find the position of the object in meters w.r.t. the camera
    """
    find_position
    Inputs: bw_target - the frame to get contours from
            frame - the frame to get the position of the rice box
            H - the homography matrix 
            bfr - the coordinates of the best fit rectangle (bounding box) for the target
    Outputs: position - a array containing the x, y and z position of the object w.r.t. to the tracker
    Description: This function find the position of the ar tag w.r.t. to the webcam. The center of the
                 image is taken is (0,0). The positive x-axis is towards the right and the positive 
                 y-axis is upwards.
    """
    def find_position(self, bw_target, frame, H, bfr):
        height, width = frame.shape[:2]
        # width of the frame
        w1 = math.fabs(bfr[3][0] - bfr[0][0])
        w2 = math.fabs(bfr[2][0] - bfr[1][0])
        width_img = (w1 + w2) / 2.0  # avg the widths of the two sides of the image
        width_of_frame=TARGET_WIDTH*width/width_img

        # height of the frame
        h1 = math.fabs(bfr[1][1] - bfr[0][1])
        h2 = math.fabs(bfr[2][1] - bfr[3][1])
        height_img = (h1 + h2) / 2.0  # avg the height of the two sides of the image
        height_of_frame=TARGET_HEIGHT*height/height_img
        
        target_h,target_w = bw_target.shape[:2]
        frame_h,frame_w = frame.shape[:2]
        #height and width of the tag w.r.t. to the resized image
        h = ar_tag_h*target_h/original_image_h
        w = ar_tag_w*target_w/original_image_w
        # finding the center of the ar tag  
        pts = np.float32([[h-1, w-1], [h-1, w + 1], [h + 1, w + 1],[h + 1, w-1]]).reshape(-1,1, 2)
        dst = HomographyandPose.perspective_transform(H, pts)
        box = np.reshape(dst,(4,2))
        box = np.int0(box)
        x,y = self.find_center(box)
        # x,y displacements of ricebox in meters from the webcam
        y = -1*(y-frame_h/2) *height_of_frame*0.0254/frame_h  
        x = (x-frame_w/2) *width_of_frame*0.0254/frame_w    #in meters
        # distance of the ricebox from the webcam in meters
        z = self.calculate_distance(frame,bfr)/3.208 #in meters
        return [x, y, z]

    """
    find_center
    Inputs: cnt- the contour that you want to find the center of
    Outputs: cx - the x coordinate of the center of the contour
            cy - the y coordinate of the center of the contour
    Description: This function finds the x, y coordinates in the frame of the center
    of the contour passed in
    """

    def find_center(self, cnt):
        M = cv2.moments(cnt)
        denom = M["m00"]
        if abs(denom) > 0.001:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = 0
            cy = 0
        return cx, cy

    """
    calculate_distance
    Inputs: frame - the frame to be analyzed
            bfr - the coordinates of the best fit rectangle (bounding box) for the target
    Outputs: the distance from the target to the camera
    Description: This function calculates the distance from the target to the camera
    """
    def calculate_distance(self, frame, bfr):
        height, width, channels = frame.shape
        w1 = math.fabs(bfr[3][0] - bfr[0][0])
        w2 = math.fabs(bfr[2][0] - bfr[1][0])
        width_img = (w1 + w2) / 2.0  # avg the widths of the two sides of the image
        w=TARGET_WIDTH*width/width_img
        distance = ((w/2)/math.tan(CAMERA_FOV*math.pi/360.0))/12.0 #in feet
        return distance

    """
    get_contours
    Inputs: frame - the frame to get contours from
    Outputs: contours - a list of contours from the frame
    Description: This function gets all of the contours that could be the target from the frame.
    It does this by converting the frame to HSV values, thresholding the hsv image to get just
    red colors, thresholding this mask so it's in black and white, and then finding the contours
    of this mask. 
    """

    def get_contours(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        kernel = np.ones((5, 5), np.uint8)

        # top right orange corner of box with webcam settings
        lower_orange = np.array([2, 74, 30])
        upper_orange = np.array([15, 190, 123])
        mask0 = cv2.inRange(hsv, lower_orange, upper_orange)

        # main red color of box with webcam settings
        lower_red = np.array([0, 126, 45])
        upper_red = np.array([180, 213, 193])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask0 + mask1

        # blend the two masks together by doing a dilation and then erosion
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        ret, thresh = cv2.threshold(closing, 127, 255, cv2.THRESH_BINARY)
        # image, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)

        cnt = contours[0]
        rect = cv2.minAreaRect(cnt)
        roi = cv2.boxPoints(rect)
        roi = np.int0(roi)

        return contours, roi

    """
    get_orientation_keypoints
    Input: frame - the frame to be analyzed
    Output: angle - the angle of rotation of the target relative to sample target image (target.py); 
                    this is None if target does not appear to be in frame
            points - an array of the x,y coordinates of matched keypoints in the frame that have matches in the 
                    sample target image; this is None if the target does not appear to be in the frame
    Description: This function determines whether or not the target is in the frame. If it is not in the frame
    (because there are too few matches during feature matching) then (None, None) is returned. Otherwise, the angle
    of rotation of the object is returned as well the x,y coordinates of keypoints that matched between this frame
    and the target sample image.
    """

    def get_keypoints(self, frame, bw_target, roi):
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [roi], 0, (255), -1)

        orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE, scaleFactor=1.2, fastThreshold=10,
                             edgeThreshold=20, patchSize=20)

        # get keypoints and descriptors for both images
        kp_target, des_target = orb.detectAndCompute(bw_target, None)
        kp_orig, des_orig = orb.detectAndCompute(grey_frame, mask)

        # create a matcher and match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_target, des_orig)

        # sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        # only allow matches where the distance between descriptors is less than MATCH_DIST
        matches = [m for m in matches if m.distance < MATCH_DIST]

        # make sure we have enough good quality matches
        if len(matches) >= MATCH_NUM_THRESHOLD:
            target_pts = np.float32([kp_target[m.queryIdx].pt for m in matches])
            orig_pts = np.float32([kp_orig[m.trainIdx].pt for m in matches])

            return orig_pts, target_pts
        else:
            return [], []

    """
    best_contour
    Inputs: contours- a list of contours that we are analyzing to see which one makes sense for our target
            keypoints - the x, y coordinates of the keypoints found that matched the target sample image
    Outputs: the one contour that is most likely to be our target. If no contour exists then return None
    Description: This function goes through all of the possible contours that could be our target and identifies
    the one most likely to be the target. It does this by considering which contours contain all or most of the
    keypoints passed in and my doing shape matching based on a sample image (target_threshold.jpg) that has
    the correct shape of the box.
    """

    def best_contour(self, contours, keypoints, desired_cnt):
        # threshold for number of keypoints that need to be inside hull
        count_threshold = 7

        # sorted the contours based on their area
        contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)

        i = 0
        for contour in contours:
            if i > 9:
                break
            else:
                hull = cv2.convexHull(contour, False)
                rect = cv2.minAreaRect(hull)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                count = self.counter(box, keypoints)

                if count < count_threshold:
                    i += 1
                    continue
                else:
                    match = cv2.matchShapes(box, desired_cnt, 1, 0.0)
                    if match <= SHAPE_MATCH_THRESHOLD:
                        return box
                    else:
                        i += 1
                        continue
        return None

    def counter(self, contour, points):
        count = 0
        # print(points)
        for p in points:
            if (cv2.pointPolygonTest(contour, (p[0], p[1]), False) >= 0):
                count = count + 1
        return count

    def get_orientation(self, bfr, orig_pts, target_pts, bw_target):
        H = HomographyandPose.RANSAC(target_pts, orig_pts)

        # if our homography function fails, try OpenCV's
        if len(H) == 0:
            H, _ = cv2.findHomography(target_pts, orig_pts, cv2.RANSAC, 5)
            if not np.any(H):
                return None, None
        # using the transformation matrix to find the 4 edges of the image
        h, w = bw_target.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = HomographyandPose.perspective_transform(H, pts)
        box = np.reshape(dst, (4, 2))
        box = np.int0(box)

        return box, H

    """
    draw_on_image
    Inputs: frame - the frame to be drawn on
            bfr - the points for the best fit rectangle
            angle - the relative rotation of the target from the sample target image (target.jpg)
            offset_angle - the relative angle of the target from the center of the image
            distance - the distance from the target to the camera
            cx - the x coordinate of the center of the target
            cy - the y coordinate of the center of the target
    Outputs: an image containing the original frame overlayed with information it
    Description: This function creates a new image that has the original frame overlayed with information. The information
    includes a bounding box around the target, text that displays the angle of rotation, offset angle, and distance
    from camera to target, and has a line from the center of the image to the center of the target.
    """

    def draw_on_image(self, frame, bfr, projected_pts):
        frame = HomographyandPose.draw(frame, bfr, projected_pts)
        cv2.drawContours(frame, [bfr], 0, (0, 255, 255), 1)
        return frame

    @staticmethod
    def get_desired_cnt(img):
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_grey, 127, 255, 0)
        # image, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours[0]
