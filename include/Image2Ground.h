#ifndef IMAGE2GROUND_H
#define IMAGE2GROUND_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include <string>
#include <eigen3/Eigen/Dense>


class Image2Ground
{
public:    
    //Image2Ground();
    ~Image2Ground();
    Image2Ground(const std::string& config_file);
    void image2ground(cv::Point2f& image_point, cv::Point2f& ground_point);

private:
    void liftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P) const;
    void backprojectSymmetric(const Eigen::Vector2d& p_u,
                                        double& theta, double& phi) const;

private:
    // intrinsic parameters
    cv::Mat k_;
    cv::Mat dist_coeffs_;

    // extrinsic parameters
    cv::Mat rotation_; 
    cv::Mat translation_;

    // left camera position
    cv::Mat camera_pos_;
};

#endif