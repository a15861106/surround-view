#ifndef SAURON_STITCH_MPBLEND_H_
#define SAURON_STITCH_MPBLEND_H_

// MPBlend.cpp : Defines the entry point for the console application.
//

//#include "mpblend.h"

//#include "pfm.h"

#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;


vector<cv::Mat> imGradFeature(cv::Mat &X);
void mergeGrad(vector<cv::Mat> &elements, int index, Mat &mask, vector<cv::Mat>& res);
void fft2(const Mat in, Mat &complexI);
Mat buildModPoissonParam(int s1, int s2);
Mat modPoisson(vector<Mat> &X, Mat &param, double ep = 1e-8, double smallpositive = 1e-12);
void avoidzero(Mat &param, double ep = 1e-8, double smallpositive = 1e-12);
void expand(Mat &in, Mat &out);
void blendWithExpanded(Mat &img);
void blendMP(vector<Mat> &img, Mat &mask, Mat &out);

void imGradFeature(cv::Mat &X, vector<cv::Mat>& feature);
void buildModPoissonParam(int s1, int s2, Mat &param);
void modPoisson(vector<Mat> &X, Mat &param, Mat& res, double ep = 1e-8, double smallpositive = 1e-12);

#endif
