import cv2 as cv
import numpy as np

def find_homography(objpts, imgpts):
    x1 = objpts[0, 0]
    y1 = objpts[0, 1]
    x2 = objpts[1, 0]
    y2 = objpts[1, 1]
    x3 = objpts[2, 0]
    y3 = objpts[2, 1]
    x4 = objpts[3, 0]
    y4 = objpts[3, 1]

    u1 = imgpts[0, 0]
    v1 = imgpts[0, 1]
    u2 = imgpts[1, 0]
    v2 = imgpts[1, 1]
    u3 = imgpts[2, 0]
    v3 = imgpts[2, 1]
    u4 = imgpts[3, 0]
    v4 = imgpts[3, 1]

    A = [[x1, y1, 1, 0, 0, 0, -u1 * x1, - u1 * y1],
         [0, 0, 0, x1, y1, 1, -v1 * x1, -v1 * y1],
         [x2, y2, 1, 0, 0, 0, -u2 * x2, -u2 * y2],
         [0, 0, 0, x2, y2, 1, -v2 *x2, -v2 * y2],
         [x3, y3, 1, 0, 0, 0, -u3 *x3, -u3 * y3],
         [0, 0, 0, x3, y3, 1, -v3 * x3, -v3 * y3],
         [x4, y4, 1, 0, 0, 0, -u4 * x4, -u4 * y4],
         [0, 0, 0, x4, y4, 1, -v4 * x4, - v4 * y4]]
    A = np.array(A, np.float32)

    b = np.array([u1, v1, u2, v2, u3, v3, u4, v4]).reshape(-1, 1)

    x = np.matmul(np.linalg.pinv(A), b)
    h33 = np.array([[1]])

    x = np.vstack((x, h33))

    H = x.reshape((3, 3))

    return H

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

    pt1 = objp[0]
    pt2 = objp[4]
    pt3 = objp[7]
    pt4 = objp[11]
    objp = np.vstack((pt1, pt2, pt3, pt4))

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img = cv.imread("../images/tutorial_checkerboard/1.png")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.undistort(gray, mtx, dist, None, mtx)

    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        corners2 = corners2.flatten().reshape(-1, 2)

        pt1 = corners2[0]
        pt2 = corners2[4]
        pt3 = corners2[7]
        pt4 = corners2[11]
        corners2 = np.vstack((pt1, pt2, pt3, pt4))

        draw_corners(img, corners2)

        H_cv, _ = cv.findHomography(objp, corners2, cv.RANSAC, 5.0)
        H = find_homography(objp, corners2)

        np.savez("tutorial_img1_H", H=H, corners=corners2, objpts=objp)

        print("Open cv homography: ", H_cv)

        print("My homography: ", H)


if __name__ == "__main__":
    main()