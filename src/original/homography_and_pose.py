import cv2 as cv
import numpy as np
import math

class HomographyandPose:
    """
    Inputs: H - a 3x3 homography matrix
            K - the intrinsic camera matrix found in the calibration step
    Outputs: T - the 3 x 4 transformation matrix (it has both rotation and translation)
             R - the 3 x 3 rotation matrix
             t - the 3 x 1 translation vector
    Description: Get the rotation matrix, translation vector, and combined transformation matrix
    from a homography matrix and camera parameters
    """
    @staticmethod
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
    @staticmethod
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
    @staticmethod
    def draw(img, corners, pts):
        pts = pts.flatten().reshape((3, 3))
        # corners = corners.flatten().reshape((42, 2))
        corner = tuple(corners[0])
        temp = tuple(pts[0, 0:2])
        print("corner", corner)
        print("pt", temp)
        img = cv.line(img, corner, tuple(pts[0, 0:2]), (255, 0, 0), 4)
        img = cv.line(img, corner, tuple(pts[1, 0:2]), (0, 255, 0), 4)
        img = cv.line(img, corner, tuple(pts[2, 0:2]), (0, 0, 255), 4)
        return img

    @staticmethod
    def perspective_transform(H, pts):
        transformed_pts = []
        for pt in pts:
            pt = np.append(pt, 1).reshape((3, 1))
            p = np.matmul(H, pt)
            transformed_pts.append([p[0] / p[2], p[1] / p[2]])
        return transformed_pts

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

    @staticmethod
    def make_subset(xy, uv):
        x = xy[0]
        y = xy[1]
        u = uv[0]
        v = uv[1]
        array = np.array([[x, y, 1, 0, 0, 0, -u * x, - u * y],
                          [0, 0, 0, x, y, 1, -v * x, -v * y]]).reshape((2, 8))
        b = np.array([[u, v]]).reshape((2, 1))
        return array, b

    """
    Inputs: objpts - a (n, 2) numpy array of points in the reference frame of the object
            imgpts - a (n, 2) numpy array of points in the reference frame of the image (
                    the pixel locations of the features)
    Returns: a 3 x 3 homography matrix to get from objpts to imgpts
    Description: calculate a homography matrix using 4 or more coplanar points
    """
    @staticmethod
    def find_homography(objpts, imgpts):
        obj_xy = objpts[0]
        img_uv = imgpts[0]
        A, b = HomographyandPose.make_subset(obj_xy, img_uv)
        for i in range(1, len(objpts)):
            obj_xy = objpts[i]
            img_uv = imgpts[i]
            sub_A, sub_b = HomographyandPose.make_subset(obj_xy, img_uv)
            A = np.vstack((A, sub_A))
            b = np.vstack((b, sub_b))

        x = np.matmul(np.linalg.pinv(A), b)
        h33 = np.array([[1]])

        x = np.vstack((x, h33))

        H = x.reshape((3, 3))

        return H

    @staticmethod
    def compute_inliers(H, obj_pts, img_pts, d):
        inliers = 0
        estimated_img_pts = HomographyandPose.perspective_transform(H, obj_pts)
        estimated_img_pts = np.array(estimated_img_pts)
        estimated_img_pts = estimated_img_pts.flatten().reshape(-1, 2)
        all_ssd = []
        obj_inliers = []
        img_inliers = []
        for i in range(np.size(img_pts, 0)):
            ssd = ((img_pts[i] - estimated_img_pts[i])**2).sum()
            all_ssd.append(ssd)
            if ssd < d:
                inliers += 1
                obj_inliers.append(obj_pts[i])
                img_inliers.append(img_pts[i])
        obj_inliers = np.array(obj_inliers)
        img_inliers = np.array(img_inliers)
        return inliers, obj_inliers, img_inliers

    @staticmethod
    def check_Collinear(points):
        threshold = 0.3
        # 1, 2, 3
        if math.fabs((points[2][1] - points[1][1]) * (points[1][0] - points[0][0]) - (points[1][1] - points[0][1]) * (
                points[2][0] - points[1][0])) < threshold:
            return False
        else:
            # 1 3 4
            if math.fabs(
                    (points[3][1] - points[2][1]) * (points[2][0] - points[0][0]) - (points[2][1] - points[0][1]) * (
                            points[3][0] - points[2][0])) < threshold:
                return False
        return True


    @staticmethod
    def RANSAC(obj_pts, img_pts):
        # number of points to be used in model
        p = 4
        # number of different points we have
        n = len(obj_pts)
        # distance threshold
        d = 5
        # ideal homography
        H = np.array([])
        # number of iterations to run ransac
        N = 2000
        # best number of inliers
        num_inliers_best = 0
        # an array of the inlier points corresponding to the best number of inliers
        best_obj_inliers = np.array([])
        best_img_inliers = np.array([])
        # number of nearby points required to assert a model fits well
        T = 15 - p
        # counter
        i = 0
        while i < N:
            # draw a sample of p points from the data
            indices = np.random.randint(0, n-1, size=p)
            obj_subset = obj_pts[indices]
            img_subset = img_pts[indices]

            # go to next iteration if points are collinear
            if not HomographyandPose.check_Collinear(obj_subset):
                continue

            # fit points
            Hk = HomographyandPose.find_homography(obj_subset, img_subset)

            # find each data point outside of sample
            outside_indices = [i for i in range(n) if obj_pts[i] not in obj_subset]
            obj_pts_outside = obj_pts[outside_indices]
            img_pts_outside = img_pts[outside_indices]

            # calculate number of inliers and the inliers
            num_inliers, obj_inliers, img_inliers = HomographyandPose.compute_inliers(Hk, obj_pts_outside, img_pts_outside, d)

            if num_inliers > num_inliers_best:
                num_inliers_best = num_inliers
                best_obj_inliers = np.vstack((obj_subset, obj_inliers))
                best_img_inliers = np.vstack((img_subset, img_inliers))
            i += 1
        print("inliers found: ", num_inliers_best)
        print("best_img_inliers shape: ", best_img_inliers.shape)
        print("best_obj_inliers shape", best_obj_inliers.shape)
        if num_inliers_best >= T:
            return HomographyandPose.find_homography(best_obj_inliers, best_img_inliers)
        else:
            return H


def draw_cv(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def main():
    """
        with np.load("tutorial_calibration.npz") as data:
        mtx, dist = [data[i] for i in ("mtx", "dist")]

    with np.load("tutorial_new_calibration.npz") as data:
        newcameramtx = data["newmtx"]


    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    print("mtx: ", mtx)
    print("new mtx:", newcameramtx)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img = cv.imread("../../images/tutorial_checkerboard/1.png")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        corners2 = corners2.flatten().reshape(-1, 2)

        H = HomographyandPose.find_homography(objp, corners2)

        T, R, t = HomographyandPose.get_rotation_matrix(H, newcameramtx)

        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).flatten().reshape((3, 3))
        projected_pts = HomographyandPose.project_pts(T, newcameramtx, axis)

        img = HomographyandPose.draw(img, corners, projected_pts)

        cv.resize(img, (200,200))
        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    """

    with np.load("homography_pts.npz") as data:
        obj_pts, img_pts, cv_homography = [data[i] for i in ("obj_pts", "img_pts", "cv_homo")]

    H = HomographyandPose.RANSAC(obj_pts, img_pts)
    print("my H", H)

    print("openCV H", cv_homography)

if __name__ == "__main__":
    main()