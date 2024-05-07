import numpy as np
import cv2
import pickle   

class VisualOdometry():
    def __init__(self, calib_dir) -> None:
        self.detector = cv2.FastFeatureDetector.create(threshold = 25, nonmaxSuppression=True)
        self.kMinNumFeature = 1500

        self.lk_params = dict(winSize  = (21, 21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        with open(calib_dir + "/camera_matrix.pkl", 'rb') as f:
            self.K = pickle.load(f)

        with open(calib_dir + "/projection_matrix.pkl", 'rb') as f:
            self.P = pickle.load(f)

        with open(calib_dir + "distortion_coefficients.pkl", 'rb') as f:
            self.D = pickle.load(f)

    def optical_flow(self, prev_img, curr_img, prev_keys):
        #sparse Lucas-Kanade optical flow algorithm
        k2, status, err = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, prev_keys, None, **self.lk_params)
        status = status.reshape(status.shape[0])

        #keep only features found in current and prev frames
        return prev_keys[status == 1], k2[status == 1]

    def get_pose(self, k1, k2):
        E, _ = cv2.findEssentialMat(k1, k2, self.K, threshold=1, method=cv2.RANSAC)
        n ,R, t, mask = cv2.recoverPose(E, k1, k2)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t

        mtx = np.matmul(self.K, T)

        return T, mtx


    def triangulate(self, mtx1, mtx2, b_prev_pts, b_cur_pts, y_prev_pts, y_cur_pts):
        b_points_3D = cv2.triangulatePoints(mtx1, mtx2, b_prev_pts, b_cur_pts)
        y_points_3D = cv2.triangulatePoints(mtx1, mtx2, y_prev_pts, y_cur_pts)

        return b_points_3D, y_points_3D
    
    def pnp(self, b_points_3D, y_points_3D, b_cur_pts, y_cur_pts):
        b_retval, b_rvec, b_tvec = cv2.solvePnP(b_cur_pts, b_points_3D, self.K, self.D)
        y_retval, y_rvec, y_tvec = cv2.solvePnP(y_cur_pts, y_points_3D, self.K, self.D)

        T_b = np.eye(4, dtype=np.float64)
        T_b[:3, :3] = b_rvec
        T_b[:3, 3] = b_tvec

        T_y = np.eye(4, dtype=np.float64)
        T_y[:3, :3] = y_rvec
        T_y[:3, 3] = y_tvec
        return T_b, T_y
    
    def process_first_frame(self, frame):
        keypoints_prev = self.detector.detect(frame)
        print(keypoints_prev)
        # keypoint detectors inherit the FeatureDetector interface
        return np.array([i.pt for i in keypoints_prev], dtype=np.float32)
    
    def process_frame(self, old_frame, new_frame, prev_keys):
        k1, k2 = self.optical_flow(old_frame, new_frame, prev_keys)
        T, mtx = self.get_pose(k1, k2)

        new_keys = self.detector.detect(new_frame)
        return np.array([i.pt for i in new_keys], dtype=np.float32), T, mtx