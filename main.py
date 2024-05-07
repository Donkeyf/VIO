import cv2 as cv
import numpy as np
from monocular_VO import VisualOdometry





def main():
    data_dir = "calibration"
    vo = VisualOdometry(data_dir)

    start_pose =  np.ones((3, 4))
    start_translation = np.zeros((3,1))
    start_rotation = np.identity(3)
    start_pose = np.concatenate((start_rotation, start_translation), axis = 1)
    curr_pose = start_pose

    bstart_pose =  np.ones((3, 4))
    bstart_translation = np.zeros((3,1))
    bstart_rotation = np.identity(3)
    bstart_pose = np.concatenate((bstart_rotation, bstart_translation), axis = 1)
    bcurr_pose = bstart_pose

    ystart_pose =  np.ones((3, 4))
    ystart_translation = np.zeros((3,1))
    ystart_rotation = np.identity(3)
    ystart_pose = np.concatenate((ystart_rotation, ystart_translation), axis = 1)
    ycurr_pose = ystart_pose

    capture = cv.VideoCapture(0)   
    capture.set(3, 128)
    capture.set(4, 96)

    old_frame = None
    new_frame = None

    bot_path = []
    b_pose = []
    y_pose = []

    prev_keys = 0

    b_pts_prev = []
    y_pts_prev = []

    mtx1 = 0


    i = 0
    while(capture.isOpened()):

        if(i == 0):
            ret, new_frame = capture.read()
            
            mtx1 = vo.P
            prev_keys = vo.process_first_frame(new_frame)
            old_frame = new_frame
            i += 1
            continue

        ret, new_frame = capture.read()
        prev_keys, transf, mtx2 = vo.process_frame(old_frame, new_frame, prev_keys)
        old_frame = new_frame

        curr_pose = np.matmul(curr_pose, np.linalg.inv(transf))
        bot_path.append((curr_pose[0, 3], 0, curr_pose[2, 3]))

        
        b_points_3D, y_points_3D = vo.triangulate(mtx1, mtx2, b_prev_pts, b_cur_pts, y_prev_pts, y_cur_pts)
        mtx1 = mtx2

        T_b, T_y = vo.pnp(b_points_3D, y_points_3D, b_cur_pts, y_cur_pts)
        
        bcurr_pose = np.matmul(bcurr_pose, np.linalg.inv(T_b))
        b_pose.append(bcurr_pose)

        ycurr_pose = np.matmul(ycurr_pose, np.linalg.inv(T_y))
        y_pose.append(ycurr_pose)