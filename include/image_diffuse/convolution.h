// Author: chernand@google.com (Carlos Hernandez)
//
// Generic code to perform a fast full convolution given the kernel class. The
// convolution is optimized for small kernels, where convolving with the full
// kernel is faster than doing two consecutive separable convolutions.
//
// Example use case extracted from GaussianHalfSize:
//
//  typedef kernel::GaussianKernel<PIXEL_TYPE, KERNEL_SIZE> KernelType;
//  typedef convolution::InnerLoop<
//    PIXEL_TYPE, KernelType, NUM_CHANNELS, KERNEL_SIZE> InnerLoopType;
//  Convolve<PIXEL_TYPE, InnerLoopType, NUM_CHANNELS, 2>(image, result);

#ifndef VISION_IMAGE_CONVOLUTION_H_
#define VISION_IMAGE_CONVOLUTION_H_

#include <algorithm>

//#include "stitch/image_diffuse/wimage.h"
#include <opencv2/opencv.hpp>


// Convolution of an image with a given kernel.
// STEP defines the pixel decimation along the x and y directions. Decimation
// happens after the convolution. If STEP > 1, the output image is decimated by
// keeping 1 pixel every STEP pixels. E.g. to implement simple convolution
// use STEP = 1. For downsampling the image by a factor of 2 use STEP = 2.
// The output width will be set to (image.Width() + STEP_X - 1) / STEP_X
// and the height to (image.Width() + STEP_Y - 1) / STEP_Y.
// INNER_LOOP is expected to expose a Call function that performs the inner
// convolution loop for one pixel.
// NUM_CHANNELS is the number of channels of the image. It must match
// image.channels().
template <typename PIXEL_TYPE, typename INNER_LOOP, int NUM_CHANNELS,
          int STEP_X, int STEP_Y>
void Convolve(const cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>& image,
              cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>* result);

// Similar to Convolve, but the output is expected to be already allocated.
// If the output size is different than the expected one, i.e.
// the size returned by Convolve above, the algorithm takes care to not
// go out of bounds in the input image or the output image.
template <typename PIXEL_TYPE, typename INNER_LOOP, int NUM_CHANNELS,
          int STEP_X, int STEP_Y>
void ConvolveNoAlloc(const cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>& image,
                     cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>* result);

// Generic implementation of DoubleSize that estimates the new pixel values
// by using a specified INNER_LOOP convolution.
// The output image will be allocated to twice the input image size.
template <typename PIXEL_TYPE, typename INNER_LOOP, int NUM_CHANNELS>
void DoubleSizeWithConvolution(
    const cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>& image,
    cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>* result);

// Similar to DoubleSizeWithConvolution, but the output image is expected
// to be already allocated. The algorithm will crop the
// result if the allocated size is less than twice the input size.
template <typename PIXEL_TYPE, typename INNER_LOOP, int NUM_CHANNELS>
void DoubleSizeWithConvolutionNoAlloc(
    const cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>& image,
    cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>* result);

namespace internal_namespace {
// Helper function to extract a patch of size KERNEL_WIDTH * KERNEL_HEIGHT
// around an input coordinate (x, y). The function deals with boundaries by
// implementing a "Clamp-To-Edge" boundary condition.
template <typename PIXEL_TYPE, int NUM_CHANNELS,
          int KERNEL_WIDTH, int KERNEL_HEIGHT>
inline void GetPatchClampedToEdge(
    const cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>& image,
    const int x,
    const int y,
    cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>* patch) {
  const int kKernelWidthHalf = (KERNEL_WIDTH - 1) / 2;
  const int kKernelHeightHalf = (KERNEL_HEIGHT - 1) / 2;
  const int max_x = image.cols - 1;//image.Width() - 1;
  const int max_y = image.rows - 1;//image.Height() - 1;
  for (int j = 0; j < KERNEL_HEIGHT; ++j) {
    for (int i = 0; i < KERNEL_WIDTH; ++i) {
      for (int c = 0; c < NUM_CHANNELS; ++c) {
        const int sample_x = std::min(max_x,
                                      std::max(0, x - kKernelWidthHalf + i));
        const int sample_y = std::min(max_y,
                                      std::max(0, y - kKernelHeightHalf + j));
        //(*patch)(i, j)[c] = image(sample_x, sample_y)[c];
        (*patch)(j, i)[c] = image(sample_y, sample_x)[c];
      }
    }
  }
}

// Helper function to extract a patch of size KERNEL_WIDTH * KERNEL_HEIGHT
// around an input coordinate (x, y). The function does not deal with left/right
// boundaries, only with top/bottom ones since they are very efficient to
// implement. The patch is copied "by reference" into an array of pointers
// for efficiency.
template <typename PIXEL_TYPE, int NUM_CHANNELS,
          int KERNEL_WIDTH, int KERNEL_HEIGHT>
inline void GetPatchRows(const cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>& image,
                         const int x,
                         const int y,
                         PIXEL_TYPE const* rows[]) {
  const int kKernelWidthHalf = (KERNEL_WIDTH - 1) / 2;
  const int kKernelHeightHalf = (KERNEL_HEIGHT - 1) / 2;
  const int height = image.rows;//image.Height();
  for (int k = 0; k < KERNEL_HEIGHT; ++k) {
    //rows[k] = image(x - kKernelWidthHalf, std::min(height - 1, std::max(0, y - kKernelHeightHalf + k)));
    rows[k] = (PIXEL_TYPE *)(image.ptr(std::min(height - 1, std::max(0, y - kKernelHeightHalf + k)), x - kKernelWidthHalf));
  }
}

inline int RoundUp(int value, int step) {
  return value + (step - (value % step)) % step;
}

inline int RoundDown(int value, int step) {
  return value - (value % step);
}

}  // namespace internal_namespace

template <typename PIXEL_TYPE, typename INNER_LOOP, int NUM_CHANNELS,
          int STEP_X, int STEP_Y>
void Convolve(const cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>& image,
              cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>* result) {
  /*CHECK_NOTNULL(result);
  CHECK_GT(image.Width(), 0);
  CHECK_GT(image.Height(), 0);*/
  // Compute the output size rounding up, e.g. half sizing a 3x3 image gives
  // an output size of 2x2.
  const int width_out = (image.cols + STEP_X - 1) / STEP_X;
  const int height_out = (image.rows + STEP_Y - 1) / STEP_Y;
  //result->Allocate(width_out, height_out);
  result->create(height_out, width_out);
  ConvolveNoAlloc<PIXEL_TYPE, INNER_LOOP, NUM_CHANNELS, STEP_X, STEP_Y>(
      image, result);
}

template <typename PIXEL_TYPE, typename INNER_LOOP, int NUM_CHANNELS,
          int STEP_X, int STEP_Y>
void ConvolveNoAlloc(const cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>& image,
                     cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>* result) {
 /* CHECK_NOTNULL(result);
  CHECK_GT(image.Width(), 0);
  CHECK_GT(image.Height(), 0);
  CHECK_GT(result->Width(), 0);
  CHECK_GT(result->Height(), 0);*/
  const int kKernelWidth = INNER_LOOP::kWidth;
  const int kKernelHeight = INNER_LOOP::kHeight;
  const int width = std::min(STEP_X * result->cols, image.cols);
  const int height = std::min(STEP_Y * result->rows, image.rows);
  const int kBoundaryWidth = kKernelWidth / 2;
  // Align the left and right boundaries to the STEP_X size.
  const int left_boundary =
      std::min(width, internal_namespace::RoundUp(kBoundaryWidth, STEP_X));
  const int right_boundary =
      std::max(left_boundary,
               internal_namespace::RoundDown(width - kBoundaryWidth, STEP_X));

  //WImageBufferC<PIXEL_TYPE, NUM_CHANNELS> temp_patch(kKernelWidth, kKernelHeight);
  cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>> temp_patch(kKernelHeight, kKernelWidth);
  const PIXEL_TYPE* patch_rows[kKernelHeight];
  for (int i = 0; i < kKernelHeight; ++i) {
    patch_rows[i] = (PIXEL_TYPE*)temp_patch.ptr(i);
  }
  const PIXEL_TYPE* rows[kKernelHeight];

  for (int y = 0; y < height; y += STEP_Y) {
    PIXEL_TYPE* res_ptr = (PIXEL_TYPE*)(result->ptr(y / STEP_Y));

    // Left block dealing with the left boundary condition.
    for (int x = 0; x < left_boundary; x += STEP_X) {
      internal_namespace::GetPatchClampedToEdge<PIXEL_TYPE, NUM_CHANNELS,
                                      kKernelWidth, kKernelHeight>(
          image, x, y, &temp_patch);
      INNER_LOOP::Call(patch_rows, res_ptr);
      res_ptr += NUM_CHANNELS;
    }

    if (left_boundary < right_boundary) {
      // Middle block, no left/right boundary conditions need to be checked.
      internal_namespace::GetPatchRows<PIXEL_TYPE, NUM_CHANNELS,
                             kKernelWidth, kKernelHeight>(
          image, left_boundary, y, rows);
      for (int x = left_boundary; x < right_boundary; x += STEP_X) {
        INNER_LOOP::Call(rows, res_ptr);
        for (int i = 0; i < kKernelHeight; ++i) {
          rows[i] += NUM_CHANNELS * STEP_X;
        }
        res_ptr += NUM_CHANNELS;
      }
    }

    // Right block dealing with the right boundary condition.
    for (int x = right_boundary; x < width; x += STEP_X) {
      internal_namespace::GetPatchClampedToEdge<PIXEL_TYPE, NUM_CHANNELS,
                                      kKernelWidth, kKernelHeight>(
          image, x, y, &temp_patch);
      INNER_LOOP::Call(patch_rows, res_ptr);
      res_ptr += NUM_CHANNELS;
    }
  }
  temp_patch.release();
}

template <typename PIXEL_TYPE, typename INNER_LOOP, int NUM_CHANNELS>
void DoubleSizeWithConvolution(
    const cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>& image,
    cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>* result) {
  CHECK_GT(image.cols, 0);
  CHECK_GT(image.rows, 0);
  CHECK_NOTNULL(result);

  //result->Allocate(image.Width() * 2, image.Height() * 2);
  result->create(image.rows * 2, image.cols * 2);
  DoubleSizeWithConvolutionNoAlloc<PIXEL_TYPE, INNER_LOOP, NUM_CHANNELS>(
      image, result);
}

// TODO(chernand): Look into merging this function and Convolve into a single
// loop dealing with convolution, downsampling and upsampling.
template <typename PIXEL_TYPE, typename INNER_LOOP, int NUM_CHANNELS>
void DoubleSizeWithConvolutionNoAlloc(
    const cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>& image,
    cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>>* result) {
  //CHECK_GT(image.Width(), 0);
  //CHECK_GT(image.Height(), 0);
  //CHECK_NOTNULL(result);
  //CHECK_GT(result->Width(), 0);
  //CHECK_GT(result->Height(), 0);

  // This logic deals with the case where the output image size is smaller than
  // 2x the input image size, i.e. we need to crop the double sized image. This
  // is often the case when building pyramids whose initial image size is not
  // a power of 2.
  const int width = std::min(result->cols, (image.cols + 1) * 2);
  const int height = std::min(result->rows, (image.rows + 1) * 2);

  // Compute the left and right boundaries.
  const int kKernelWidth = INNER_LOOP::kWidth;
  const int kKernelHeight = INNER_LOOP::kHeight;
  const int kBoundaryWidth = kKernelWidth / 2 + (kKernelWidth / 2) % 2;
  const int left_boundary = std::min(width + width % 2, kBoundaryWidth);
  const int right_boundary =
      std::max(left_boundary, width + width % 2 - kBoundaryWidth);
  //WImageBufferC<PIXEL_TYPE, NUM_CHANNELS> temp_patch(kKernelWidth, kKernelHeight);
  cv::Mat_<cv::Vec<PIXEL_TYPE, NUM_CHANNELS>> temp_patch(kKernelHeight, kKernelWidth);
  const PIXEL_TYPE* patch_rows[kKernelHeight];
  for (int i = 0; i < kKernelHeight; ++i) {
    //patch_rows[i] = temp_patch(0, i);
    //patch_rows[i] = temp_patch(i, 0);
    patch_rows[i] = (PIXEL_TYPE* )temp_patch.ptr(i);
  }
  const PIXEL_TYPE* rows[kKernelHeight];

  for (int y = 0; y < height; ++y) {
    const bool row_is_odd = y % 2;
    PIXEL_TYPE* res_ptr = (PIXEL_TYPE*)(result->ptr(y));

    // Left block dealing with the left boundary condition.
    for (int x = 0; x < left_boundary; x += 2) {
      internal_namespace::GetPatchClampedToEdge<PIXEL_TYPE, NUM_CHANNELS,
                                      kKernelWidth, kKernelHeight>(
          image, x / 2, y / 2, &temp_patch);
      const bool output_two_samples = x < width - 1;
      INNER_LOOP::Call(patch_rows, row_is_odd, output_two_samples, res_ptr);
      res_ptr += NUM_CHANNELS * 2;
    }

    if (left_boundary < right_boundary) {
      // Middle block, no boundary conditions need to be checked.
      internal_namespace::GetPatchRows<PIXEL_TYPE, NUM_CHANNELS,
                             kKernelWidth, kKernelHeight>(
          image, left_boundary / 2, y / 2, rows);
      for (int x = left_boundary; x < right_boundary; x += 2) {
        INNER_LOOP::Call(rows, row_is_odd, res_ptr);
        res_ptr += NUM_CHANNELS * 2;
        for (int i = 0; i < kKernelHeight; ++i) {
          rows[i] += NUM_CHANNELS;
        }
      }
    }

    // Right block dealing with the right boundary condition.
    for (int x = right_boundary; x < width; x += 2) {
      internal_namespace::GetPatchClampedToEdge<PIXEL_TYPE, NUM_CHANNELS,
                                      kKernelWidth, kKernelHeight>(
          image, x / 2, y / 2, &temp_patch);
      const bool output_two_samples = x < width - 1;
      INNER_LOOP::Call(patch_rows, row_is_odd, output_two_samples, res_ptr);
      res_ptr += output_two_samples ? NUM_CHANNELS * 2 : NUM_CHANNELS;
    }
  }
  temp_patch.release();
}

#endif  // VISION_IMAGE_CONVOLUTION_H_

