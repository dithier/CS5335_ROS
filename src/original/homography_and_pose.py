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
Inputs: objpts - a (n, 2) numpy array of points in the reference frame of the object
        imgpts - a (n, 2) numpy array of points in the reference frame of the image (
                the pixel locations of the features)
Returns: a 3 x 3 homography matrix to get from objpts to imgpts
Description: calculate a homography matrix using 4 or more coplanar points
"""
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

    x = np.matmul(np.linalg.pinv(A), b)
    h33 = np.array([[1]])

    x = np.vstack((x, h33))

    H = x.reshape((3, 3))

    return H

"""
Inputs: xy - a (1,2) numpy array of a point in the reference frame of the object
        uv - a (1,2) numpy array of a point in the reference frame of the image (the pixel 
            location of the feature)
Returns: array - a (2, 8) numpy array that is part of the array for calculations for the homography
            matrix
        b = a (2, 1) numpy array containing the point in the reference frame of the image
Description: compute subarrays of the two matrices used to calculate the homography matrix in the
find_homography function
"""
def make_subset(xy, uv):
    x = xy[0]
    y = xy[1]
    u = uv[0]
    v = uv[1]
    array = np.array([[x, y, 1, 0, 0, 0, -u * x, - u * y],
                      [0, 0, 0, x, y, 1, -v * x, -v * y]]).reshape((2, 8))
    b = np.array([[u, v]]).reshape((2, 1))
    return array, b

def draw_cv(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def main():
    with np.load("tutorial_calibration.npz") as data:
        mtx, dist = [data[i] for i in ("mtx", "dist")]

    with np.load("tutorial_new_calibration.npz") as data:
        newcameramtx = data["newmtx"]

    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    print("mtx: ", mtx)
    print("new mtx:", newcameramtx)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img = cv.imread("../../images/tutorial_checkerboard/2.png")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        corners2 = corners2.flatten().reshape(-1, 2)

        H = find_homography(objp, corners2)

        T, R, t = get_rotation_matrix(H, newcameramtx)

        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).flatten().reshape((3, 3))
        projected_pts = project_pts(T, newcameramtx, axis)

        img = draw(img, corners, projected_pts)

        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()