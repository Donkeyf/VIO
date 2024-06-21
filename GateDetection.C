#include <jevois/Core/Module.H>
#include <jevois/Debug/Log.H>
#include <jevois/Util/Utils.H>
#include <jevois/Image/RawImageOps.H>
#include <jevois/Debug/Timer.H>
#include <jevois/Util/Coordinates.H>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Geometry> // for AngleAxis and Quaternion

static jevois::ParameterCategory const ParamCateg("GateDetection Options");

//! Parameter \relates GateDetection
JEVOIS_DECLARE_PARAMETER(objsize, cv::Size_<float>, "Object size (in meters)",
                         cv::Size_<float>(1.5F, 1.5F), ParamCateg);

//! Parameter \relates GateDetection
JEVOIS_DECLARE_PARAMETER(camparams, std::string, "File stem of camera parameters, or empty. Camera resolution "
			 "will be appended, as well as a .yaml extension. For example, specifying 'calibration' "
			 "here and running the camera sensor at 320x240 will attempt to load "
			 "calibration320x240.yaml from within directory " JEVOIS_SHARE_PATH "/camera/",
			 "calibration", ParamCateg);


class GateDetection : public jevois::StdModule,
                    public jevois::Parameter<objsize>
    {
    protected:
        cv::Mat itsCamMatrix;
        cv::Mat itsDistCoeffs;


    class SinglePoseEstimationParallel : public cv::ParallelLoopBody
    {
      public:
        SinglePoseEstimationParallel(cv::Mat & _objPoints, cv::InputArrayOfArrays _corners,
                                     cv::InputArray _cameraMatrix, cv::InputArray _distCoeffs,
                                     cv::Mat & _rvecs, cv::Mat & _tvecs) :
            objPoints(_objPoints), corners(_corners), cameraMatrix(_cameraMatrix),
            distCoeffs(_distCoeffs), rvecs(_rvecs), tvecs(_tvecs)
        { }

        void operator()(cv::Range const & range) const
        {
          int const begin = range.start;
          int const end = range.end;
          
          for (int i = begin; i < end; ++i)
            cv::solvePnP(objPoints, corners.getMat(i), cameraMatrix, distCoeffs,
                         rvecs.at<cv::Vec3d>(i), tvecs.at<cv::Vec3d>(i));
        }

        private:
        cv::Mat & objPoints;
        cv::InputArrayOfArrays corners;
        cv::InputArray cameraMatrix, distCoeffs;
        cv::Mat & rvecs, tvecs;

    };


    public:
    
    //Constructor
    GateDetection(std::string const & instance): jevois::StdModule(instance)
    {}

    virtual ~GateDetection() {}

    //Pose estimation 6D
    void estimatePose(std::vector<std::vector<cv::Point2f> > & corners, cv::OutputArray _rvecs,
                      cv::OutputArray _tvecs)
    {
        auto const box = objsize::get();

        // set coordinate system in the middle of the object, with Z pointing out
        cv::Mat objPoints(4, 1, CV_32FC3);
        objPoints.ptr< cv::Vec3f >(0)[0] = cv::Vec3f(-box.width * 0.5F, -box.height * 0.5F, 0);
        objPoints.ptr< cv::Vec3f >(0)[1] = cv::Vec3f(-box.width * 0.5F, box.height * 0.5F, 0);
        objPoints.ptr< cv::Vec3f >(0)[2] = cv::Vec3f(box.width * 0.5F, box.height * 0.5F, 0);
        objPoints.ptr< cv::Vec3f >(0)[3] = cv::Vec3f(box.width * 0.5F, -box.height * 0.5F, 0);
    
        int ngates = (int)corners.size();
        rvecs.create(ngates, 1, CV_64FC3); _tvecs.create(ngates, 1, CV_64FC3);
        cv::Mat rvecs = _rvecs.getMat(), tvecs = _tvecs.getMat();
        cv::parallel_for_(cv::Range(0, nobj), SinglePoseEstimationParallel(objPoints, corners, itsCamMatrix,
                                                                           itsDistCoeffs, rvecs, tvecs));
      
    }

    //load camera calibration
    void loadCameraCalibration(unsigned int w, unsigned int h)
    {
      camparams::freeze();
      
      std::string const cpf = std::string(JEVOIS_SHARE_PATH) + "/camera/" + camparams::get() +
        std::to_string(w) + 'x' + std::to_string(h) + ".yaml";
      
      cv::FileStorage fs(cpf, cv::FileStorage::READ);
      if (fs.isOpened())
      {
        fs["camera_matrix"] >> itsCamMatrix;
        fs["distortion_coefficients"] >> itsDistCoeffs;
        LINFO("Loaded camera calibration from " << cpf);
      }
      else
      {
        LERROR("Failed to read camera parameters from file [" << cpf << "] -- IGNORED");
        itsCamMatrix = cv::Mat::eye(3, 3, CV_64F);
        itsDistCoeffs = cv::Mat::zeros(5, 1, CV_64F);
      }   
    }
}