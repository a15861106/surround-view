#ifndef CALIBRATE_H
#define CALIBRATE_H

//calibrate four cameras

#include "opencv2/opencv.hpp"
#include <vector>
#include <string>

#include "CameraPos.h"
#include "chessboard/Chessboard.h"
#include "ceres/ceres.h"

struct CalibrateOptions
{
    std::string camera_params_file; // intrinsic params file
    cv::Size chessboard_size; // default (9, 6)
    cv::Size output_size; // output image size

    double car_x;//2.2;
    double car_y;//4.0;
    double left_right_to_front_distance;
    double viewrange; // 左右可视范围

    CalibrateOptions()
    {
        camera_params_file = "../config/glsl.yml";
        chessboard_size = cv::Size(6, 4);
        output_size = cv::Size(600, 600);
        car_x = 2.193;
        car_y = 5.117;
        left_right_to_front_distance = 1.5;
        viewrange = 20; //10.5 12 20 18 左右可视范围        
    }
};

class Calibrate
{

public:
    Calibrate(CalibrateOptions options);
    ~Calibrate();

    bool readCamParam(const std::string &camera_file);

    bool run(std::vector<cv::Mat>& inputs);

    void undistortImage(cv::Mat& input, cv::Mat& output, CAMERA_POS pos, bool recompute=false);

    // find corner points in chessboard image, default use opencv method
    bool findCornerPoints(cv::Mat& image, std::vector<cv::Point2f>& corners, CAMERA_POS pos, bool use_opencv=true);

    // generate correspoing points in top-down image
    bool generateProjectedPoints(std::vector<cv::Point2f>& undistorted_image_points, std::vector<cv::Point2f>& image_points, std::vector<cv::Point2f>& projected_points,
                                     CAMERA_POS pos, cv::Size undistort_image_size);

    bool generateHomography(cv::Mat& undistort_image, cv::Mat& output, std::vector<cv::Point2f>& image_points, std::vector<cv::Point2f>& projected_points,
                                     CAMERA_POS pos, cv::Size outputSize, cv::Mat_<double>& H);

    void optimizeHomography(std::vector<std::vector<cv::Point2f>>& image_points, std::vector<std::vector<cv::Point2f>>& projected_points);

    std::vector<cv::Mat_<double>>& getHomography(){return homography_matrixs_;};
    CalibrateOptions& getOptions(){return options_;};
    cv::Size getCameraSize() {return camera_size_;};
    
    void estimateExtrinsics(const std::vector<cv::Point3f>& objectPoints,
                                    const std::vector<cv::Point2f>& imagePoints,
                                    cv::Mat& rvec, cv::Mat& tvec, CAMERA_POS pos) const;
    void liftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P, CAMERA_POS pos) const;
    void backprojectSymmetric(const Eigen::Vector2d& p_u,
                                        double& theta, double& phi, CAMERA_POS pos) const;
    void spaceToPlane(const cv::Point3f p3, cv::Point2f& p, cv::Mat& rvec,
                             cv::Mat& tvec, CAMERA_POS pos) const;
    void optimizationCameraParameters(std::vector<std::vector<cv::Point2f>>& image_points, 
                                   std::vector<std::vector<cv::Point2f>>& projected_points);
    
    std::vector<cv::Mat>& getRoatationVectors() {return rotation_vectors_;}
    std::vector<cv::Mat>& getTranslationVectors() {return translation_vectors_;}
    std::vector<cv::Mat>& getK() {return k_;}
    std::vector<cv::Mat>& getDist() {return dist_coeffs_;}

    void optimizationExtrinsicParameters(std::vector<std::vector<cv::Point2f>>& image_points, 
                                   std::vector<std::vector<cv::Point2f>>& projected_points);     

    void testProject(const cv::Point3f& P_, cv::Point2f& p, CAMERA_POS pos);
    std::vector<cv::Mat>& getRemapX();
    std::vector<cv::Mat>& getRemapY();
    cv::Size getUndistortImageSize();
    
    void outputParams(std::string& file_name);
    void setCameraSize(cv::Size camera_size) {camera_size_ = camera_size;}
    void setRemapX(std::vector<cv::Mat>& x) {mapx_ = x;}
    void setRemapY(std::vector<cv::Mat>& y) {mapy_ = y;}
    void setHomography(std::vector<cv::Mat_<double>>& h) {homography_matrixs_ = h;}
    void setR(std::vector<cv::Mat>& r) {rotation_vectors_ = r;}
    void setT(std::vector<cv::Mat>& t) {translation_vectors_ = t;}
    void setRemap();

private:
    // check whether quad is black square
    bool isChessboardSquare(std::vector<cv::Point2f>& image_points, std::vector<cv::Point>& square_point, 
            int image_height, int image_width, int num, CAMERA_POS pos);
    void preprocessSquarePoint(std::vector<cv::Point>& points, CAMERA_POS pos);

    void visualizeCornerPoints(cv::Mat& image, std::vector<cv::Point2f>& image_points, CAMERA_POS pos);

    CalibrateOptions options_;
    cv::Size camera_size_; // input camera frame size

    std::vector<cv::Mat_<double>> homography_matrixs_; // homography matrix
    std::vector<cv::Mat> k_; // calibration matrix
    std::vector<cv::Mat> dist_coeffs_; //
    std::vector<cv::Mat> mapx_;
    std::vector<cv::Mat> mapy_;
    std::vector<cv::Mat> new_k_;
    int multi_camera_parameters_;

    std::vector<cv::Mat> undistort_images_;

    // extrinsic params
    std::vector<cv::Mat> rotation_vectors_;
    std::vector<cv::Mat> translation_vectors_;

    std::vector<cv::Mat> inputs_;
};


#endif