import cv2 as cv
import numpy as np

def find_homography(objpts, imgpts):
    obj_xy = objpts[0]
    img_uv = imgpts[0]
    A, b = make_subset(obj_xy, img_uv)
    for i in range(1, len(objpts)):
        obj_xy = objpts[i]
        img_uv = imgpts[i]
        sub_A, sub_b = make_subset(obj_xy, img_uv)
        A = np.vstack((A, sub_A))
        b = np.vstack((b, sub_b))


    # np.savez("old_calcs", A=A,b=b)

    x = np.matmul(np.linalg.pinv(A), b)
    h33 = np.array([[1]])

    x = np.vstack((x, h33))

    H = x.reshape((3, 3))

    return H

def make_subset(xy, uv):
    x = xy[0]
    y = xy[1]
    u = uv[0]
    v = uv[1]
    array = np.array([[x, y, 1, 0, 0, 0, -u * x, - u * y],
                      [0, 0, 0, x, y, 1, -v * x, -v * y]]).reshape((2, 8))
    b = np.array([[u, v]]).reshape((2, 1))
    return array, b

def draw_corners(img, corners):
    for corner in corners:
        cv.circle(img, (corner[0], corner[1]), 6, (0, 0, 255), -1)
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def main():
    with np.load("tutorial_calibration.npz") as data:
        mtx, dist = [data[i] for i in ("mtx", "dist")]

    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    """
     pt1 = objp[0]
    pt2 = objp[4]
    pt3 = objp[7]
    pt4 = objp[11]
    objp = np.vstack((pt1, pt2, pt3, pt4))
    """

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img = cv.imread("../../images/tutorial_checkerboard/7.png")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    u_gray = cv.undistort(gray, mtx, dist, None, mtx)

    ret, corners = cv.findChessboardCorners(u_gray, (7, 6), None)
    if ret == True:
        corners2 = cv.cornerSubPix(u_gray, corners, (11, 11), (-1, -1), criteria)
        corners2 = corners2.flatten().reshape(-1, 2)

        """
        pt1 = corners2[0]
        pt2 = corners2[4]
        pt3 = corners2[7]
        pt4 = corners2[11]
        corners2 = np.vstack((pt1, pt2, pt3, pt4))
        """

        draw_corners(img, corners2)

        H_cv, _ = cv.findHomography(objp, corners2, cv.RANSAC, 5.0)
        H = find_homography(objp, corners2)

        # np.savez("tutorial_img1_H", H=H, corners=corners2, objpts=objp)

        print("Open cv homography: ", H_cv)

        print("My homography: ", H)


if __name__ == "__main__":
    main()