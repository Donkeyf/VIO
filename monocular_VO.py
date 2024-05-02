import numpy as np
import cv2
import pickle   

class VisualOdometry():
    def __init__(self, calib_dir) -> None:
        self.R = 0
        self.T = 0

        self.detector = cv2.FastFeatureDetector.create(threshold = 25, nonmaxSuppression=True)
        self.kMinNumFeature = 1500

        self.lk_params = dict(winSize  = (21, 21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        with open(calib_dir + "/camera_matrix.pkl", 'rb') as f:
            self.K = pickle.load(f)

        with open(calib_dir + "/projection_matrix.pkl", 'rb') as f:
            self.P = pickle.load(f)

    def optical_flow(self, prev_img, curr_img, prev_keys):
        #sparse Lucas-Kanade optical flow algorithm
        k2, status, err = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, prev_keys, None, **self.lk_params)
        status = status.reshape(status.shape[0])

        #keep only features found in current and prev frames
        return prev_keys[status == 1], k2[status == 1]

    def get_pose(self, k1, k2):
        E, _ = cv2.findEssentialMat(k1, k2, self.K, threshold=1, method=cv2.RANSAC)
        n ,R, T, mask = cv2.recoverPose(E, k1, k2)

        self.R = np.dot(R, self.R)
        self.T = self.T + scale*(self.R*T)