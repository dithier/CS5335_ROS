import yaml
import numpy as np
import cv2
from tracking_class import Tracking

def main():
    # get camera matrix from calibration file
    path = "./"
    filename = path + "webcam.yaml"

    with open(filename, "r") as file_handle:
        calib_data = yaml.load(file_handle)

    cameramtx = calib_data["camera_matrix"]["data"]
    cameramtx = np.array(cameramtx).reshape((3,3))

    # load image in question
    img = cv2.imread("../../images/tracking2/2.jpeg")
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

    # process the frame
    tracking_obj = Tracking(bw_target, desired_cnt, cameramtx)

    # show results
    annotated_frame, quaternion, position = tracking_obj.process_frame(img)
    print("quaternion: ", quaternion)
    print("position: ", position)

    cv2.imshow('image', annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
