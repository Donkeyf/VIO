import numpy as np
import cv2
import pickle 

class VisualOdometry_3D2D():
    def __init__(self, calib_dir) -> None:
        #Fast Feature Detector Setup
        self.orb = cv2.ORB_create(3000)
        self.kMinNumFeature = 1500


        with open(calib_dir + "/camera_matrix.pkl", 'rb') as f:
            self.K = pickle.load(f)

        with open(calib_dir + "/projection_matrix.pkl", 'rb') as f:
            self.P = pickle.load(f)

        with open(calib_dir + "distortion_coefficients.pkl", 'rb') as f:
            self.D = pickle.load(f)

    def undistort_img(self, img):
        h, w = img.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w, h), 1, (w, h))
        new_img = cv2.undistort(img, self.K, self.D, None, new_mtx)

        return new_img

    def feature_detection(self, img):
        kp, des = self.orb.detectAndCompute(img, None)
        return kp, des

    def feature_match(self, img, des, kp):
        kp2, des2 = self.orb.detectAndCompute(img, None)
        match = self.flann.knnMatch(des, des2, k=2)

        filter = []
        try:
            for m, n in match:
                if m.distance < 0.8 * n.distance:
                    filter.append(m)
        except ValueError:
            pass

        q1 = np.float32([kp[m.queryIdx].pt for m in filter])
        q2 = np.float32([kp2[m.trainIdx].pt for m in filter])
        return q1, q2

    def triangulate(self, q1, q2):
        points_4D = cv2.triangulatePoints(self.P, self.P, q1, q2)
        points_3D = points_4D / points_4D[3]
        points_3D = points_3D[:3].T

        return points_3D

    def get_pose(self, img, q, points_3D):
        #needs 3 points for q and points_3D
        retval, R, t = cv2.solveP3P(q, points_3D, self.K, self.D)
