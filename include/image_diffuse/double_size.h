// Double size implementations with different kernels.

#ifndef SAURON_STITCH_IMAGE_DIFFUSE_DOUBLE_SIZE_H_
#define SAURON_STITCH_IMAGE_DIFFUSE_DOUBLE_SIZE_H_

//#include "stitch/image_diffuse/wimage.h"
#include "convolution.h"
#include "convolution_loop.h"
#include "kernel.h"
#include <opencv2/opencv.hpp>


// Double size implementation using a bi-linear kernel.
// The output image is expected to be allocated. The algorithm will crop the
// result if the allocated size is less than twice the input size.
template <typename PIXEL_TYPE, int NUM_CHANNELS>
void BiLinearDoubleSizeNoAlloc(const cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>& image,
                               cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>* result) {
  typedef BiLinearKernelGroup<PIXEL_TYPE> KernelType;
  typedef InnerLoopWithGroup<
    PIXEL_TYPE, KernelType, NUM_CHANNELS, 3, 3> InnerLoopType;
  DoubleSizeWithConvolutionNoAlloc<PIXEL_TYPE, InnerLoopType, NUM_CHANNELS>(
      image, result);
}

// Double size implementation using a bi-linear kernel.
// The output image will be allocated to twice the input image size.
template <typename PIXEL_TYPE, int NUM_CHANNELS>
void BiLinearDoubleSize(const cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>& image,
                        cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>* result) {
  typedef BiLinearKernelGroup<PIXEL_TYPE> KernelType;
  typedef InnerLoopWithGroup<
    PIXEL_TYPE, KernelType, NUM_CHANNELS, 3, 3> InnerLoopType;
  DoubleSizeWithConvolution<PIXEL_TYPE, InnerLoopType, NUM_CHANNELS>(
      image, result);
}

// Double size implementation using a bi-linear kernel that only updates
// pixels whose alpha channel is already 0. The alpha channel is assumed
// to be the last channel of the image.
// The output image is expected to be allocated. The algorithm will crop the
// result if the allocated size is less than twice the input size.
template <typename PIXEL_TYPE, int NUM_CHANNELS>
void BiLinearDoubleSizeWithMaskNoAlloc(
    const cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>& image,
    cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>* result) {
  typedef BiLinearKernelGroup<PIXEL_TYPE> KernelType;
  typedef MaskedInnerLoopWithGroup<
    PIXEL_TYPE, KernelType, NUM_CHANNELS, 3, 3> InnerLoopType;
  DoubleSizeWithConvolutionNoAlloc<PIXEL_TYPE, InnerLoopType, NUM_CHANNELS>(
      image, result);
}

// Double size implementation using a bi-linear kernel that only updates
// pixels whose alpha channel is already 0. The alpha channel is assumed
// to be the last channel of the image.
// The output image will be allocated to twice the input image size.
template <typename PIXEL_TYPE, int NUM_CHANNELS>
void BiLinearDoubleSizeWithMask(
    const cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>& image,
    cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>* result) {
  typedef BiLinearKernelGroup<PIXEL_TYPE> KernelType;
  typedef MaskedInnerLoopWithGroup<
    PIXEL_TYPE, KernelType, NUM_CHANNELS, 3, 3> InnerLoopType;
  DoubleSizeWithConvolution<PIXEL_TYPE, InnerLoopType, NUM_CHANNELS>(
      image, result);
}


#endif  // VISION_IMAGE_DOUBLE_SIZE_H_

