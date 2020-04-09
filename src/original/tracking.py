"""
This module is responsible for tracking a red box object. It finds the object, puts
a best fit rectangle bounding box around the object and finds the objects relative orientation
to the sample image (target.jpg). It also calculates the objects offset angle from the
center of the camera and the distance from the camera. It displays all of this information
on a new frame.
"""
from homography import find_homography
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
# from __future__ import division

# camera field of view
CAMERA_FOV = 55

# actual width (in inches) of target
TARGET_WIDTH = 9.25

# actual height (in inches) of target
TARGET_HEIGHT = 11.25

# the maximum allowed distance for matches
MATCH_DIST = 30

# the minimum number of keypoint matches there has to be to identify the target
MATCH_NUM_THRESHOLD = 10

# the number of keypoints to be used to find the best contour
KEYPOINT_NUM = 25

# threshold for matching shapes to confirm box match
SHAPE_MATCH_THRESHOLD = 0.1

# min distance that keypoints have to be apart
RADIUS = 50

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
def process_frame(frame, bw_target, desired_cnt):
    frame_copy = np.copy(frame)
    # get contours
    contours = get_contours(frame_copy)
    # get rotation angle of target
    orig_pts, target_pts = get_keypoints(frame, bw_target,contours,desired_cnt)
    bfr = best_contour(contours, orig_pts, desired_cnt)

    """
    cv2.drawContours(img_copy, [bfr], 0, (0, 255, 0), 5)
    cv2.imshow("bfr", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """


    if np.any(bfr):
        angle, box = get_orientation(bfr, orig_pts, target_pts, bw_target)

        if angle is not None:

            print("angle", angle)

            """
            cv2.drawContours(frame, [bfr], 0, (0, 255, 255), 2)
            cv2.imshow('BFR', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """
            # find coordinates of center of contour
            cx, cy = find_center(box)

            # find the angle offset the target is from center of camera
            offset_angle = get_offset_angle(frame, cx)

            # find the distance of the camera to the target
            distance = calculate_distance(frame, box) # carter

            # annotate and draw extra information onto the frame
            #frame = draw_on_image(frame, bfr, angle, offset_angle, distance, cx, cy) # vinay

            frame = draw_on_image(frame, box, angle, offset_angle, distance, cx, cy)  # vinay
        else:
            print("Homography could not be found")
    else:
        print("no valid contours")

    return frame


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


def get_keypoints(frame, bw_target,contours,desired_cnt):
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)

    # get keypoints and descriptors for both images
    kp_target, des_target = orb.detectAndCompute(bw_target, None)
    kp_orig, des_orig = orb.detectAndCompute(grey_frame, None)

    # create a matcher and match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_target, des_orig)

    # sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    # only allow matches where the distance between descriptors is less than MATCH_DIST
    matches = [m for m in matches if m.distance < MATCH_DIST]
    """
     img3 = cv2.drawMatches(bw_target, kp_target, frame, kp_orig, matches[:10], None, flags=2)

    plt.imshow(img3), plt.show()
    """

    if len(matches) > 0:
        print("last match distance", matches[-1].distance)

    print("num matches: ", len(matches))

    # make sure we have enough good quality matches
    if len(matches) >= MATCH_NUM_THRESHOLD:
        target_pts = np.float32([kp_target[m.queryIdx].pt for m in matches])
        orig_pts = np.float32([kp_orig[m.trainIdx].pt for m in matches])

        """
                for pt in orig_pts:
            cv2.circle(img_copy, tuple(pt), 20, (0, 255, 0), -1)

            cv2.imshow("pt", img_copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        """
        return orig_pts, target_pts
    return None, None

def distance(pt1, pt2):
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** (1/2)


def not_in_radius(pt, prev_pts):
    for prev_pt in prev_pts:
        print("prev: ", prev_pt)
        print("pt: ", pt)
        print("distance: ", distance(pt, prev_pt))
        if distance(pt, prev_pt) < RADIUS:
            return False
    return True

def get_orientation(bfr, orig_pts, target_pts, bw_target):
    """
    for pt in orig_pts:
        cv2.circle(img_copy, tuple(pt), 10, (0, 255, 0), -1)

    cv2.drawContours(img_copy, [bfr], 0, (0, 255, 255), 5)

    cv2.imshow("4 points", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """




	## code to check if the target points are collinear and exist in the best contour
    img_points = []
    obj_points = []
    count =0
    i = 0
    for i in range(len(target_pts)):
        img_p = orig_pts[i]
        obj_p = target_pts[i]

        """
         cv2.circle(img_copy, tuple(img_p), 20, (0, 255, 0), -1)

        cv2.imshow("pt", img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        #print("p",p)
        if(counter(bfr, [img_p]) == 1):
            #if cv2.pointPolygonTest(bfr, (p[0], p[1]), False) >= 0:
            #count = count + 1
            img_points.append(img_p)
            obj_points.append(obj_p)

            """
                        if count == 4:
                if (check_Collinear(img_points)):
                    break
                else:
                    img_points.pop(0)
                    img_points.pop(0)
                    obj_points.pop(0)
                    obj_points.pop(0)
                    count = count - 2
            
            
        if(i == len(target_pts) - 1):
            print("No non-collinear point found")
            return None, None
            """

    #"""

    for pt in img_points:
        cv2.circle(img_copy, tuple(pt), 10, (0, 255, 0), -1)

    cv2.drawContours(img_copy, [bfr], 0, (0, 255, 255), 5)

    cv2.imshow("4 points", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #"""
    if len(obj_points) < 4:
        return None, None
    # M = find_homography(np.array(img_points),np.array(obj_points))
    M = find_homography(np.array(obj_points), np.array(img_points))
    print("My H: ", M)




    # H, mask = cv2.findHomography(target_pts, orig_pts, cv2.RANSAC, 5)
    H , mask = cv2.findHomography(np.array(obj_points), np.array(img_points), cv2.RANSAC, 5)
    print("CV2 H:", H)
    # matchesMask = mask.ravel().tolist()

    ## using the transformation matrix to find the 4 edges of the image
    h, w = bw_target.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],[w - 1, 0]]).reshape(-1,1, 2)
    dst = cv2.perspectiveTransform(pts, H)
    box = np.reshape(dst,(4,2))
    box = np.int0(box)
    print("box: ", box)

    b = M[0][1];
    a = M[0][0];
    # angle to get from box in frame to 0 degrees
    angle =  -math.atan2(b, a) * (180 / math.pi)
    # multiply by -1 to get from 0 degrees to frame (the box in the frame's angle of rotation)
    angle = -1 * round(angle, 0)

    if angle < 0:
        angle += 360

    """
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv2.drawMatches(bw_target, kp_target, grey_frame, kp_orig, matches, None, **draw_params)
    plt.imshow(img3, 'gray'), plt.show()
    """
    return angle, box

"""Inputs: points - an array of 4 points.
Output: Boolean value - true if it's collinear
Description: This function checks whether the points passed are collinear. It return True if the points are not collinear
    
"""
def check_Collinear(points):
    #1, 2, 3
    if math.fabs((points[2][1] - points[1][1]) * (points[1][0] - points[0][0]) - (points[1][1] - points[0][1]) * (points[2][0] - points[1][0]))<.1:
        return False
    else:
        #1 3 4
        if math.fabs((points[3][1] - points[2][1]) * (points[2][0] - points[0][0]) - (points[2][1] - points[0][1]) * (points[3][0] - points[2][0]))<0.1:
            return False
    return True


"""
get_offset_angle
Inputs: frame - the frame to be analyzed
        cx - the x coordinate for the center of the target
Outputs: angle- the angle from the target to the center of the frame
Description; This function finds the offset angle of the target relative to the
center of the frame.
"""
def get_offset_angle(frame, cx):
    height, width, channels = frame.shape
    offset = (width / 2) - cx
    angle = (CAMERA_FOV * (offset / width))
    return angle

"""
calculate_distance
Inputs: frame - the frame to be analyzed
        bfr - the coordinates of the best fit rectangle (bounding box) for the target
Outputs: the distance from the target to the camera
Description: This function calculates the distance from the target to the camera
"""
def calculate_distance(frame, bfr):
    height, width, channels = frame.shape

    w1 = math.fabs(bfr[3][0] - bfr[0][0])
    w2 = math.fabs(bfr[2][0] - bfr[1][0])
    width_img = (w1 + w2) / 2.0  # avg the widths of the two sides of the image

    w=TARGET_WIDTH*width/width_img
    distance = ((w/2)/math.tan(CAMERA_FOV*math.pi/360.0))/12.0 #in feet
    print('distance: ',distance)
    return distance

"""
find_center
Inputs: cnt- the contour that you want to find the center of
Outputs: cx - the x coordinate of the center of the contour
        cy - the y coordinate of the center of the contour
Description: This function finds the x, y coordinates in the frame of the center
of the contour passed in
"""
def find_center(cnt):
    M = cv2.moments(cnt)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy

def rotate_point(point, M):
    return [np.matmul(M, point[0])]

def get_rotation_matrix(angle):
    angle = angle * (math.pi / 180)
    M = np.array([[math.cos(angle), -math.sin(angle)],
                 [math.sin(angle), math.cos(angle)]])
    return M

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
def best_contour(contours, keypoints, desired_cnt):
    """
    Identify one contour that is most likely to be our target:
    - Sort contours from largest to smallest (go through first 2-5 biggest contours?)
    - might need to do convex hull to get the full box shape
    - can do things like match shape: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
        - for match shape will need to make black and white mask image with right aspect ratio (see box dimensions above); you can generate this image yourself using draw functions
        - for speed it will probably be best to extract the contours of the shape in main() and pass that contour in process_frame and this function so we aren't redoing the coontour extraction each frame
    - check aspect ratio (although matchshapes might be enough and we may not have to do this)
    - and make sure all keypoints reside inside the contour (https://stackoverflow.com/questions/50670326/how-to-check-if-point-is-placed-inside-contour)
    """
    """
    
    cntsSorted = sorted(contours, key=lambda x: len(x), reverse=True)
    cnt = cntsSorted[0]
    cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 6)
   
    cx, cy = find_center(desired_cnt)
    desired_cnt = desired_cnt - [cx, cy]
    M = get_rotation_matrix(-angle)
    result = map(lambda x: rotate_point(x, M), desired_cnt)
    rotated_cnt = np.array(list(result), dtype=np.int)
    rotated_cnt += [cx, cy]
    cv2.drawContours(desired, [rotated_cnt], 0, (0, 255, 0), 3)
    cv2.imshow('rotated image', desired)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    # threshold for number of keypoints that need to be inside hull
    count_threshold = round(0.4 * len(keypoints), 0)
    print("count threshold: ", count_threshold)

    ## sorted the contours based on their area
    contours  = sorted(contours, key=lambda contour:cv2.contourArea(contour),reverse=True)

    i = 0
    for contour in contours:
        if i > 9:
            break
        else:
            hull = cv2.convexHull(contour, False)
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            count = counter(box, keypoints)

            """ 
            cv2.drawContours(img_copy, [box], 0, (0, 255, 255), 2)
            cv2.drawContours(img_copy, [desired_cnt], 0, (255, 0, 0), 2)
            cv2.imshow('best_contour', img_copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """

            print("count: ", count)
            if count < count_threshold:
                i += 1
                continue
            else:
                # match = cv2.matchShapes(box, rotated_cnt, 1, 0.0)
                match = cv2.matchShapes(box, desired_cnt, 1, 0.0)
                print("match cnt: ", match)
                if match <= SHAPE_MATCH_THRESHOLD:
                    return box
                else:
                    i += 1
                    continue

    ## sorting the contours based on the number of kepypoints in them
    #ctrs = sorted(contours,key = lambda count:counter(count,keypoints),reverse=True )
    #cnt = ctrs[0]

    # print(cv2.contourArea(contours[0]))
    # epsilon = 0.000000000001 * cv2.arcLength(cnt, True)
    # approx = cv2.approxPolyDP(cnt, epsilon, True)
    #
    # rect = cv2.minAreaRect(cnt)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # img = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    #cv2.drawContours(img, approx, -1, (0, 255, 0), 3)
    #cv2.imshow('hi', img)

    ## convhull to get better contour
    """
    hull = cv2.convexHull(cnt, returnPoints=False)
    ## finding the defects in the contour and fixing them
    defects = cv2.convexityDefects(cnt, hull)
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        cv2.line(img, start, end, [0, 255, 0], 2)
    """
    """
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print (keypoints)
    """

    return None

def counter(contour,points):
    count = 0

    """
    cv2.drawContours(img, [contour], 0, (0, 255, 0), 6)
 
        for point in points:
        cv2.circle(img, (point[0], point[1]), 15, (0, 255, 255), -1)
    cv2.imshow('counter', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    #print(points)
    for p in points:
        if(cv2.pointPolygonTest(contour, (p[0], p[1]), False)>=0):
            count=count+1
    return count
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
def draw_on_image(frame, bfr, angle, offset_angle, distance, cx, cy):
    """
    Draw bounding box on image
    Above bounding box say Object Rotation : angle degrees
    In bottom right corner of frame put offset_angle and distance
    Draw line from center of image to center of target
    """
    cv2.drawContours(frame, [bfr], 0, (0, 255, 0), 5)
    return frame

"""
get_contours
Inputs: frame - the frame to get contours from
Outputs: contours - a list of contours from the frame
Description: This function gets all of the contours that could be the target from the frame.
It does this by converting the frame to HSV values, thresholding the hsv image to get just
red colors, thresholding this mask so it's in black and white, and then finding the contours
of this mask. 
"""
def get_contours(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    kernel = np.ones((9, 9), np.uint8)

    lower_orange = np.array([21, 67, 149])
    upper_orange = np.array([95, 255, 255])
    mask0 = cv2.inRange(hsv, lower_orange, upper_orange)

    # main red color of box
    lower_red = np.array([165, 97, 58])
    upper_red = np.array([188, 216, 235])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # increase edges that are not thick at bottom of box
    # mask1 = cv2.dilate(mask1, kernel, iterations=2)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel, iterations=1)

    # deep red color of box
    """
    lower_deep_red = np.array([0, 0, 0])
    upper_deep_red = np.array([139, 255, 255])
    mask2 = cv2.inRange(hsv, lower_deep_red, upper_deep_red)
    mask2 = cv2.bitwise_not(mask2)
    
    mask = mask0 + mask1 + mask2
    """

    mask = mask0 + mask1

    # blend the two masks together by doing a dilation and then erosion
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


    ret, thresh = cv2.threshold(closing, 127, 255, cv2.THRESH_BINARY)
    # image, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda contour:cv2.contourArea(contour), reverse=True)
    """
    cv2.imshow('orange', mask0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('red', mask1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('combined mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('dilation', closing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cnt = contours[0]
    hull = cv2.convexHull(cnt, False)
    cv2.drawContours(frame, [hull], 0, (0,255,0), 6)
    cv2.imshow('convex hull', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(frame, [box], 0, (0, 255, 255), 2)
    cv2.imshow('BFR', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cnt = contours[0]
    hull = cv2.convexHull(cnt, False)
    cv2.drawContours(frame, [hull], 0, (0,255,0), 6)
    cv2.imshow('convex hull', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(frame, [box], 0, (0, 255, 255), 2)
    cv2.imshow('BFR', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    return contours

def get_desired_cnt(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_grey, 127, 255, 0)
    # image, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]

def main():
    # test image

    img = cv2.imread("../../images/test/270.jpg")
    global img_copy
    img_copy = np.copy(img)

    # global desired
    desired = cv2.imread("../../images/desired_cnt.png")
    desired_cnt = get_desired_cnt(desired)

    # image that has target in it at 0 degree rotation
    bw_target = cv2.imread("../../images/1.jpg", 0)
    # contours = get_contours(img)

    annotated_image = process_frame(img, bw_target, desired_cnt)
    # contours = get_contours(bw_target)
    # best_contour(contours)
    annotated_image = cv2.resize(annotated_image,(600,600))
    cv2.imshow('image', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
