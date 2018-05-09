#ifndef SAURON_STITCH_IMAGE_DIFFUSE_HALF_SIZE_H_
#define SAURON_STITCH_IMAGE_DIFFUSE_HALF_SIZE_H_

//#include "stitch/image_diffuse/wimage.h"
#include "convolution.h"
#include "convolution_loop.h"
#include "kernel.h"
#include <opencv2/opencv.hpp>


// Half size implementation where the input image is first convolved with a
// 2x2 box kernel.
template <typename PIXEL_TYPE, int NUM_CHANNELS>
void BoxHalfSize(const cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>& image,
                 cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>* result);

// Half size implementation where the input image is first convolved with a
// 2x2 box kernel. The output is expected to be allocated to the right size.
template <typename PIXEL_TYPE, int NUM_CHANNELS>
void BoxHalfSizeNoAlloc(const cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>& image,
                        cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>* result);

// Half size implementation where the input image is first convolved with a
// Gaussian kernel. Current suported kernel sizes are 3x3, 4x4 and 5x5.
template <typename T, int NUM_CHANNELS, int KERNEL_SIZE>
void GaussianHalfSize(const cv::Mat_<cv::Vec<T, NUM_CHANNELS>>& image,
                      cv::Mat_<cv::Vec<T, NUM_CHANNELS>>* result);

// Half size implementation where the input image is first convolved with a
// Gaussian kernel. Current suported kernel sizes are 3x3, 4x4 and 5x5.
// The output is expected to be allocated to the right size.
template <typename T, int NUM_CHANNELS, int KERNEL_SIZE>
void GaussianHalfSizeNoAlloc(const cv::Mat_<cv::Vec<T, NUM_CHANNELS>>& image,
                             cv::Mat_<cv::Vec<T, NUM_CHANNELS>>* result);

// Half size implementation along the x-axis where the input image is first
// convolved with a Gaussian kernel. Current suported kernel sizes are
// 3x1, 4x1 and 5x1.
// The output is expected to be allocated to the right size.
template <typename T, int NUM_CHANNELS, int KERNEL_SIZE>
void GaussianHalfSizeHorizontal(const cv::Mat_<cv::Vec<T, NUM_CHANNELS>>& image,
                                cv::Mat_<cv::Vec<T, NUM_CHANNELS>>* result);

// Half size implementation along the y-axis where the input image is first
// convolved with a Gaussian kernel. Current suported kernel sizes are
// 1x3, 1x4 and 1x5.
// The output is expected to be allocated to the right size.
template <typename T, int NUM_CHANNELS, int KERNEL_SIZE>
void GaussianHalfSizeVertical(const cv::Mat_<cv::Vec<T, NUM_CHANNELS>>& image,
                              cv::Mat_<cv::Vec<T, NUM_CHANNELS>>* result);

// -------------------------- implementation ------------------------------

template <typename PIXEL_TYPE, int NUM_CHANNELS>
void BoxHalfSize(const cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>& image,
                 cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>* result) {
 /* CHECK_NOTNULL(result);*/
  //result->Allocate((image.Width() + 1) / 2, (image.Height() + 1) / 2);
  result->create((image.rows + 1) / 2, (image.cols + 1) / 2);
  BoxHalfSizeNoAlloc<PIXEL_TYPE, NUM_CHANNELS>(image, result);
}

template <typename PIXEL_TYPE, int NUM_CHANNELS>
void BoxHalfSizeNoAlloc(const cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>& image,
                        cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>* result) {
  /*CHECK_NOTNULL(result);
  CHECK(!result->IsNull());
  CHECK((result->Width() == (image.Width() + 1) / 2) ||
        (result->Width() == image.Width() / 2));
  CHECK(result->Height() == (image.Height() + 1) / 2 ||
        (result->Height() == image.Height() / 2));*/

  const int kKernelSize = 2;
  const int kStep = 2;
  typedef BoxKernel<PIXEL_TYPE, kKernelSize, kKernelSize> KernelType;
  typedef InnerLoop<
    PIXEL_TYPE, KernelType, NUM_CHANNELS, kKernelSize, kKernelSize>
      InnerLoopType;
  ConvolveNoAlloc<PIXEL_TYPE, InnerLoopType, NUM_CHANNELS, kStep, kStep>(
      image, result);
}

template <typename T, int NUM_CHANNELS, int KERNEL_SIZE>
void GaussianHalfSize(const cv::Mat_<cv::Vec<T, NUM_CHANNELS>>& image,
                      cv::Mat_<cv::Vec<T, NUM_CHANNELS>>* result) {
  CHECK_NOTNULL(result);
  //result->Allocate((image.Width() + 1) / 2, (image.Height() + 1) / 2);
  result->create((image.rows + 1) / 2, (image.cols + 1) / 2);
  GaussianHalfSizeNoAlloc<T, NUM_CHANNELS, KERNEL_SIZE>(image, result);
}

template <typename T, int NUM_CHANNELS, int KERNEL_SIZE>
void GaussianHalfSizeNoAlloc(const cv::Mat_<cv::Vec<T, NUM_CHANNELS>>& image,
                             cv::Mat_<cv::Vec<T, NUM_CHANNELS>>* result) {
 /* CHECK_NOTNULL(result);
  CHECK(!result->IsNull());
  CHECK((result->Width() == (image.Width() + 1) / 2) ||
        (result->Width() == image.Width() / 2));
  CHECK(result->Height() == (image.Height() + 1) / 2 ||
        (result->Height() == image.Height() / 2));*/

  const int kStep = 2;
  if (KERNEL_SIZE < 5) {
    typedef GaussianKernel<T, KERNEL_SIZE, KERNEL_SIZE> KernelType;
    typedef InnerLoop<
      T, KernelType, NUM_CHANNELS, KERNEL_SIZE, KERNEL_SIZE> InnerLoopType;
    ConvolveNoAlloc<T, InnerLoopType, NUM_CHANNELS, kStep, kStep>(
        image, result);
  } else {
    typedef GaussianKernel<T, KERNEL_SIZE, 1> kKernelX;
    typedef GaussianKernel<T, 1, KERNEL_SIZE> kKernelY;
    typedef InnerLoop<
      T, kKernelX, NUM_CHANNELS, KERNEL_SIZE, 1> InnerLoopX;
    typedef InnerLoop<
      T, kKernelY, NUM_CHANNELS, 1, KERNEL_SIZE> InnerLoopY;
    cv::Mat_<cv::Vec<T, NUM_CHANNELS>> temp(image.rows, image.cols/kStep);//(image.Width() / kStep, image.Height());
    ConvolveNoAlloc<T, InnerLoopX, NUM_CHANNELS, kStep, 1>(image, &temp);
    ConvolveNoAlloc<T, InnerLoopY, NUM_CHANNELS, 1, kStep>(temp, result);
  }
}

template <typename T, int NUM_CHANNELS, int KERNEL_SIZE>
void GaussianHalfSizeHorizontal(const cv::Mat_<cv::Vec<T, NUM_CHANNELS>>& image,
                                cv::Mat_<cv::Vec<T, NUM_CHANNELS>>* result) {
 /* CHECK_NOTNULL(result);
  CHECK(!result->IsNull());
  CHECK((result->Width() == (image.Width() + 1) / 2) ||
        (result->Width() == image.Width() / 2));*/

  const int kStep = 2;
  typedef GaussianKernel<T, KERNEL_SIZE, 1> kKernelX;
  typedef InnerLoop<
    T, kKernelX, NUM_CHANNELS, KERNEL_SIZE, 1> InnerLoopX;
  ConvolveNoAlloc<T, InnerLoopX, NUM_CHANNELS, kStep, 1>(image, result);
}

template <typename T, int NUM_CHANNELS, int KERNEL_SIZE>
void GaussianHalfSizeVertical(const cv::Mat_<cv::Vec<T, NUM_CHANNELS>>& image,
                              cv::Mat_<cv::Vec<T, NUM_CHANNELS>>* result) {
 /* CHECK_NOTNULL(result);
  CHECK(!result->IsNull());
  CHECK(result->Height() == (image.Height() + 1) / 2 ||
        (result->Height() == image.Height() / 2));*/

  const int kStep = 2;
  typedef GaussianKernel<T, 1, KERNEL_SIZE> kKernelY;
    typedef InnerLoop<
      T, kKernelY, NUM_CHANNELS, 1, KERNEL_SIZE> InnerLoopY;
  ConvolveNoAlloc<T, InnerLoopY, NUM_CHANNELS, 1, kStep>(image, result);
}


#endif  // SAURON_STITCH_IMAGE_DIFFUSE_HALF_SIZE_H_
