import cv2 as cv
import numpy as np
import math

"""
Inputs: H - a 3x3 homography matrix
        K - the intrinsic camera matrix found in the calibration step
Outputs: T - the 3 x 4 transformation matrix (it has both rotation and translation)
         R - the 3 x 3 rotation matrix 
         t - the 3 x 1 translation vector
Description: Get the rotation matrix, translation vector, and combined transformation matrix
from a homography matrix and camera parameters
"""
def get_rotation_matrix(H, K):
    H = np.matmul(np.linalg.pinv(K), H)

    norm = math.sqrt(H[0,0]**2 + H[1,0]**2 + H[2,0]**2)

    H = H / norm

    rc1 = H[:, 0].reshape((3,1))
    rc2 = H[:, 1].reshape((3,1))
    rc3 = np.cross(rc1, rc2, axis=0)
    t = H[:, 2].reshape((3,1))

    R = np.hstack((rc1, rc2, rc3))
    T = np.hstack((R, t))

    return T, R, t

"""
Inputs: T - the transformation matrix in form [R | t] where R is a rotation matrix and t
        is the translation vector. It is of size (3, 4)
        K - the intrinsic camera calibration matrix determined in the camera calibration step
        pts - the pts that you want projected from 3D space into 2D space
Outputs: a numpy array where all input pts have been projected onto the 2D plane of the image
"""
def project_pts(T, K, pts):
    projected = []
    for pt in pts:
        pt = np.append(pt, 1)
        p1 = np.matmul(K, T)
        p = np.matmul(p1, pt)
        p = p / p[-1]
        p = p.astype(int)
        projected.append(p)
    return np.array(projected)

"""
Inputs: img - the img you want to draw lines on
        corners - a list of corners where the 0th element is the pixel location for the origin
            of the axis
        pts - a (3,3) numpy array that has three points that when connected to the origin
            create the x, y, z axis
Outputs: an image with x, y, z axis in red, green, blue drawn on it
Description: Given an origin (corners[0]) and three points that have been projected from 3D space
to the 2D plane, draw x, y, z axis
"""
def draw(img, corners, pts):
    pts = pts.flatten().reshape((3, 3))
    corners = corners.flatten().reshape((42, 2))
    corner = tuple(corners[0])
    img = cv.line(img, corner, tuple(pts[0, 0:2]), (255, 0, 0), 5)
    img = cv.line(img, corner, tuple(pts[1, 0:2]), (0, 255, 0), 5)
    img = cv.line(img, corner, tuple(pts[2, 0:2]), (0, 0, 255), 5)
    return img

"""
Inputs: objpts - a (4, 2) numpy array of points in the reference frame of the object
        imgpts - a (4, 2) numpy array of points in the reference frame of the image (
                the pixel locations of the features)
Returns: a 3 x 3 homography matrix to get from objpts to imgpts
Description: calculate a homography matrix using 4 coplanar points
"""
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
    img = cv.imread("../images/tutorial_checkerboard/2.png")
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

        H = find_homography(objp, corners2)

        T, R, t = get_rotation_matrix(H, mtx)

        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).flatten().reshape((3, 3))
        projected_pts = project_pts(T, mtx, axis)

        img = draw(img, corners, projected_pts)

        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()