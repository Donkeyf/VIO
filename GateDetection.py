import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois
import cv2
import numpy as np
import math

class GateDetection:

    #load comera calibration from jevois share directory
    def load_camera_calibration(self, w, h):
        cpf = pyjevois.share + "/camera/calibration{}x{}.yaml".format(w, h)
        fs = cv2.FileStorage(cpf, cv2.FILE_STORAGE_READ)
        if (fs.isOpened()):
            self.camMatrix = fs.getNode("camera_matrix").mat()
            self.distCoeffs = fs.getNode("distortion_coefficients").mat()
            jevois.LINFO("Loaded camera calibration from {}".format(cpf))
        else:
            jevois.LERROR("Failed to read camera parameters from file [{}] -- IGNORED".format(cpf))
            self.camMatrix = np.eye(3, 3, dtype=double)
            self.distCoeffs = np.zeros(5, 1, dtype=double)

    
    def detect(self):
        pass

    def estimate_pose(self, gates):
        rvecs = []
        tvecs = []

        #set coordinate system in middle of objects, z pointing out
        obj_points = np.array([ ( -self.owm * 0.5, -self.ohm * 0.5, 0 ),
                               ( -self.owm * 0.5,  self.ohm * 0.5, 0 ),
                               (  self.owm * 0.5,  self.ohm * 0.5, 0 ),
                               (  self.owm * 0.5, -self.ohm * 0.5, 0 ) ])
        

        for gate_corners in gates:
            gate_reshape = np.array(gate_corners, dtype=np.float).reshape(4,2,1)
            
            ok, rv, tv = cv2.solveP3P(obj_points, gate_reshape, self.camMatrix, self.distCoeffs)

            if ok:
                rvecs.append(rv)
                tvecs.append(tv)
            else:
                rvecs.append(np.array([ (0.0), (0.0), (0.0) ]))
                tvecs.append(np.array([ (0.0), (0.0), (0.0) ]))

        return (rvecs, tvecs)   
    

    def send_serial(self, rvecs, tvecs):
        axis = rvecs
        angle = (axis[0] ** 2 + axis[1] **2 + axis[2]**2) * 0.5

        # This code lifted from pyquaternion from_axis_angle:
        mag_sq = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]
        if (abs(1.0 - mag_sq) > 1e-12): 
            axis = axis / (mag_sq ** 0.5)
        theta = angle / 2.0
        r = math.cos(theta)
        i = axis * math.sin(theta)  

        jevois.sendSerial("{} {} {} {} {} {}".
                          format(np.asscalar(tvecs[0]), np.asscalar(tvecs[1]), np.asscalar(tvecs[2]), 
                                np.asscalar(i[0]), np.asscalar(i[1]), np.asscalar(i[2])))


    def processNoUSB(self, inframe):

        if not hasattr(self, 'camMatrix'): self.load_camera_calibration(w, h)

        img = inframe.getCvBGR()
        h, w, _ = img.shape

        gates = self.detect(img)

        rvecs, tvecs = self.estimate_pose(img)



        
