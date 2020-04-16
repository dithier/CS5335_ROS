"""
This file is to create a trackbar with a picture so we can figure out what HSV bounds should used to get a good
mask image that isolates the box we are tracking in the video.

This file may not be needed. First in the tracking.py file we should try:
red = np.uint8([[[0,0,255 ]]])
hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)

and then for our upper and lower bounds we use: take [H-10, 100,100] and [H+10, 255, 255] as lower bound and
upper bound respectively

If this does not work well for our purposes, then we should write this file
"""

# make trackbar like: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_trackbar/py_trackbar.html#trackbar
# sliders should be H_low, H_high, S_low, S_high, V_low, V_high
# window should display sliders and a predefined image
# when switch is off original image should be shown
# when switch is on the mask of the image should be shown
# the mask should use code like this but using the slider HSV values: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces

import cv2
import numpy as np

def nothing(x):
    pass

def main():
    mode = "orange"

    source = "webcam"
    """
    # orange
    color = np.uint8([[[0, 165, 255]]])
    """
    if source == "phone":
        if mode == "orange":
            """
            lower_orange = np.array([15, 67, 149])
            upper_orange = np.array([95, 255, 255])
            """
            h_low_start = 0
            s_low_start = 79
            v_low_start = 42
            h_high_start = 16
            s_high_start = 126
            v_high_start = 150
        elif mode == "red":
            """
            # 3_tiles
            h_low_start = 164
            s_low_start = 70
            v_low_start = 74
            h_high_start = 178
            s_high_start = 255
            v_high_start = 215
            
            # non-tiles
            lower_red = np.array([160, 50, 50])
            upper_red = np.array([255, 255, 255])
            """
            h_low_start = 166
            s_low_start = 77
            v_low_start = 58
            h_high_start = 188
            s_high_start = 216
            v_high_start = 235
        else:
            """
            lower_deep_red = np.array([0, 0, 0])
            upper_deep_red = np.array([139, 255, 255])
            
            H: [0, 168]
            S: [0, 255]
            V : [0, 255]
            """
            h_low_start = 0
            s_low_start = 0
            v_low_start = 0
            # h_high_startt = 60
            h_high_start = 139
            s_high_start = 255
            v_high_start = 255
    elif source == "webcam":
        if mode == "orange":
            """
            lower_orange = np.array([15, 67, 149])
            upper_orange = np.array([95, 255, 255])
            """
            h_low_start = 9
            s_low_start = 92
            v_low_start = 88
            h_high_start = 19
            s_high_start = 124
            v_high_start = 142
        elif mode == "red":
            """
            h_low_start = 0
            s_low_start = 150
            v_low_start = 45
            h_high_start = 255
            s_high_start = 205
            v_high_start = 127
            """
            h_low_start = 5
            s_low_start = 115
            v_low_start = 62
            h_high_start = 215
            s_high_start = 191
            v_high_start = 144


    cv2.namedWindow('image')
    img = cv2.imread("../../images/tracking2/0.jpeg")
    # img = cv2.imread("../../images/calibration/0.jpeg")
    #img = cv2.imread("../../images/test/135_tiles.png")
    height = 300
    h, w = img.shape[:2]
    r = height / float(h)
    dim = (int(w * r), height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    orig_img = np.copy(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # create trackbars for color change
    cv2.createTrackbar('H_low', 'image', 0, 255, nothing)
    cv2.createTrackbar('H_high', 'image', 0, 255, nothing)
    cv2.createTrackbar('S_low', 'image', 0, 255, nothing)
    cv2.createTrackbar('S_high', 'image', 0, 255, nothing)
    cv2.createTrackbar('V_low', 'image', 0, 255, nothing)
    cv2.createTrackbar('V_high', 'image', 0, 255, nothing)

    # set trackbars for initial starting point
    cv2.setTrackbarPos('H_low', 'image', h_low_start)
    cv2.setTrackbarPos('H_high', 'image', h_high_start)
    cv2.setTrackbarPos('S_low', 'image', s_low_start)
    cv2.setTrackbarPos('S_high', 'image', s_high_start)
    cv2.setTrackbarPos('V_low', 'image', v_low_start)
    cv2.setTrackbarPos('V_high', 'image', v_high_start)

    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'image', 0, 1, nothing)
    cv2.setTrackbarPos(switch, 'image', 1)

    while (1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        # if ESC key pressed
        if k == 27:
            break

        # get current positions of four trackbars
        h_low = cv2.getTrackbarPos('H_low', 'image')
        h_high = cv2.getTrackbarPos('H_high', 'image')
        s_low = cv2.getTrackbarPos('S_low', 'image')
        s_high = cv2.getTrackbarPos('S_high', 'image')
        v_low = cv2.getTrackbarPos('V_low', 'image')
        v_high = cv2.getTrackbarPos('V_high', 'image')

        print("H: ", [h_low, h_high])
        print("S: ", [s_low, s_high])
        print("V: ", [v_low, v_high])

        s = cv2.getTrackbarPos(switch, 'image')

        if s == 0:
            img = orig_img
        else:
            lower_bound = np.array([h_low, s_low, v_low])
            upper_bound = np.array([h_high, s_high, v_high])
            img = cv2.inRange(hsv, lower_bound, upper_bound)

            if mode == "deep_red":
                img = cv2.bitwise_not(img)

    print("H: ", [h_low, h_high])
    print("S: ", [s_low, s_high])
    print("V: ", [v_low, v_high])
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()