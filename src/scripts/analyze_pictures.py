import cv2
from tracking_class import Tracking
import yaml
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from math import cos, sin, pi

def create_tracking_obj():
    # get camera matrix from calibration file
    path = "./"
    filename = path + "webcam.yaml"

    with open(filename, "r") as file_handle:
        calib_data = yaml.load(file_handle)

    cameramtx = calib_data["camera_matrix"]["data"]
    cameramtx = np.array(cameramtx).reshape((3, 3))

    # load image in question
    img = cv2.imread("../../images/tracking2/10.jpeg")
    height = 450
    h, w = img.shape[:2]
    r = height / float(h)
    dim = (int(w * r), height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # get desired contour for perfect match
    desired = cv2.imread("../../images/desired_cnt.png")
    desired_cnt = Tracking.get_desired_cnt(desired)

    # load image that has target in it at 0 degree rotation
    bw_target = cv2.imread("../../images/2.jpg", 0)
    h, w = bw_target.shape[:2]
    r = height / float(h)
    dim = (int(w * r), height)
    bw_target = cv2.resize(bw_target, dim, interpolation=cv2.INTER_AREA)

    # get tracking obj
    tracking_obj = Tracking(bw_target, desired_cnt, cameramtx)
    return tracking_obj

def main():
    results = []

    path = "/Users/ithier/Documents/CS5335/Final_Project/CS5335_ROS/results/"

    tracking_obj = create_tracking_obj()

    with np.load("50_images.npz") as data:
        images = data["images"]

    n = len(images)
    i = 0
    try:
        for i in range(n):
            print("i: ", i)
            img = images[i]
            annotated_frame, q, p = tracking_obj.process_frame(img)

            # record results
            results.append([i, p[0], p[1], p[2], q[0], q[1], q[2], q[3]])

            img_name = path + str(i) + ".jpg"
            cv2.imwrite(img_name, annotated_frame)

        df = pd.DataFrame(results, columns = ["Image number", "pos_x", "pos_y", "pos_z",
                                              "o_x", "o_y", "o_z", "o_w"])
        filename = path + "Our_results.csv"
        df.to_csv(filename, index=False)
    except:
        print("stopped on img ", i)
        df = pd.DataFrame(results, columns=["Image number", "pos_x", "pos_y", "pos_z",
                                            "o_x", "o_y", "o_z", "o_w"])
        filename = path + "Our_results.csv"
        df.to_csv(filename, index=False)

if __name__ == "__main__":
    main()