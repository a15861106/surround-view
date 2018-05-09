#ifndef COMPOSITION_H
#define COMPOSITION_H

#include "opencv2/opencv.hpp"
#include <vector>
#include <boost/shared_ptr.hpp>
#include "Calibrate.h"
#include <string>

typedef boost::shared_ptr<Calibrate> CalibratePtr;
typedef boost::shared_ptr<const Calibrate> CalibrateConstPtr;

class Composition
{
public:
    Composition(CalibratePtr cal);
    Composition(const std::string& param_file);
    void setCompositionByCalibrate(CalibratePtr cal);

    ~Composition();

    CalibratePtr calibrate(void);

    const CalibrateConstPtr calibrate(void) const;

    cv::Mat run(std::vector<cv::Mat>& inputs);

    void generateCompositionMask();

    cv::Mat adjustToneByGains(std::vector<cv::Mat>& inputs);

    cv::Mat adjustToneByPoisson(std::vector<cv::Mat>& inputs);

    void generateGains(std::vector<cv::Mat>& inputs);

    // using rvecs and tvecs to generate topview results
    void InitRemapMatrixs();
    void generateTopViewByImp(std::vector<cv::Mat>& inputs);

    void image2ground(cv::Point2f& image_point, cv::Point2f& ground_point);

    void camera2ground(cv::Point2f image_point, CAMERA_POS pos, cv::Point2f& ground_point);

private:
    CalibratePtr calibrate_;
    cv::Mat composition_mask_;
    std::vector<cv::Mat> camera_masks_;
    std::vector<double> gains_;
    std::vector<cv::Mat> blender_masks_;

    //params for poisson blending
    std::vector<cv::Mat> x_;
    std::vector<cv::Mat> grad_;
    cv::Mat param_;

    float start_w_, end_w_, start_h_, end_h_;

    //diffusion mask of left camera
    cv::Mat2f diffuse_mask_;
    void generateDiffueMask();
  
    //
    std::vector<cv::Mat> remap_matrixs_;
    std::vector<cv::Rect> remap_rects_;
    std::vector<cv::Mat> remap_masks_;

    // for image2ground
    std::vector<cv::Mat> rotation_matrixs_;
    std::vector<cv::Mat> translation_vectors_;
    // left camera position
    cv::Mat camera_pos_;
    cv::Mat sv_to_image_;

    cv::Size outputSize;
    std::vector<cv::Mat_<double>> homography_matrixs;
	cv::Mat res;
	cv::Mat output;         
};

#endif