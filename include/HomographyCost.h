#ifndef HOMOGRAPHY_COST_H_
#define HOMOGRAPHY_COST_H_

#include "ceres/ceres.h"
#include "opencv2/opencv.hpp"
#include "vector"

/*
use projection or reprojection to check whether ceres optimization can 
improve homography transform
*/

class HomographyCost {
 public:
  bool operator()(const double* H, double* residuals) const;
  // Create a numerically differentiated CostFunction object using
  // this functor.
  static ceres::CostFunction* Create(double x,
                                     double y,
                                     double px,
                                     double py);
  ~HomographyCost(){ };
 private:
  // Private constructor to ensure that this class can only be
  // instantiated by calling the factory Create.
  HomographyCost() {}

  double observed_x;
  double observed_y;

  double point_x;
  double point_y;
};

class GlobalCost {
public:
  bool operator()(const double* H, double* residuals) const;
  static ceres::CostFunction* Create(double x,
                                     double y,
                                     double px,
                                     double py, int pos, bool flag);

  static ceres::CostFunction* Create(double x,
                                     double y,
                                     double x2, double y2,
                                     double px,
                                     double py, int pos, bool flag);                                     
  ~GlobalCost(){};

private:
  GlobalCost() {};
  double observed_x;
  double observed_y;

  double observed_x2;
  double observed_y2;

  double point_x;
  double point_y;

  bool flag;
  int pos;
};

// this cost is used for optimization of intrinsic and extrinsic parameters
class  CameraCost{
public:
  bool operator()(const double* K, const double* H, double* residuals) const;

  static ceres::CostFunction* Create(cv::Point3f sp, 
                                   cv::Point2f ip, int pos);                                     
  ~CameraCost(){};

private:
  CameraCost() {};
  
  cv::Point3f space_point;
  cv::Point2f image_point;

  int pos;
};

// This cost is used for optimization of extrinsic parameters only
class  ExtrinsicCost{
public:
  bool operator()(const double* H, double* residuals) const;

  static ceres::CostFunction* Create(std::vector<cv::Mat> k_matrix, cv::Point3f sp, 
                                   cv::Point2f ip, int pos);                                     
  ~ExtrinsicCost(){};

private:
  ExtrinsicCost() {};
  
  cv::Point3f space_point;
  cv::Point2f image_point;
  std::vector<cv::Mat> k;

  int pos;
};
#endif