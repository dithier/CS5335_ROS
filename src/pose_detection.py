import cv2 as cv
import numpy as np
import math

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

def draw_cv(img, corners, imgpts):
    corner = tuple(corners[0])
    pt1 = tuple(imgpts[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def draw(img, corners, pts):
    pts = pts.flatten().reshape((3, 3))
    pt = tuple(pts[0, 0:2])
    corner = tuple(corners[0])
    img = cv.line(img, corner, tuple(pts[0, 0:2]), (255, 0, 0), 5)
    img = cv.line(img, corner, tuple(pts[1, 0:2]), (0, 255, 0), 5)
    img = cv.line(img, corner, tuple(pts[2, 0:2]), (0, 0, 255), 5)
    return img

def main():
    with np.load("tutorial_img1_H.npz") as data:
        H, corners, objpts = [data[i] for i in ("H", "corners", "objpts")]

    with np.load("tutorial_calibration.npz") as data:
        mtx, dist = [data[i] for i in ("mtx", "dist")]

    T, R, t = get_rotation_matrix(H, mtx)

    my_rvec, _ = cv.Rodrigues(T[:3, :3])
    print("my rcvec", my_rvec)
    print("my t", t)

    #print("My T:", T)

    # Find the rotation and translation vectors.
    ret, rvecs, tvecs = cv.solvePnP(objpts, corners, mtx, dist)
    R_cv, _ = cv.Rodrigues(rvecs)

    print("cv R:", rvecs)
    print("cv t:", tvecs)

    # project 3D points to image plane
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).flatten().reshape((3,3))
    imgpts, _ = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

    my_pts = project_pts(T, mtx, axis)

    img = cv.imread("../images/tutorial_checkerboard/1.png")
    img = cv.undistort(img, mtx, dist, None, mtx)

    img = draw_cv(img, corners, imgpts)
    #img = draw(img, corners, my_pts)

    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()