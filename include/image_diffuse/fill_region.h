#ifndef SAURON_STITCH_IMAGE_DIFFUSE_FILL_REGION_H_
#define SAURON_STITCH_IMAGE_DIFFUSE_FILL_REGION_H_

#include <math.h>
#include <algorithm>
#include <memory>
#include <vector>

#include "macros.h"
//#include "stitch/image_diffuse/wimage.h"
#include "double_size.h"
#include "half_size.h"
#include <opencv2/opencv.hpp>


// Performs a region filling operation on an alpha-premultiplied image.
// The region to be filled is specified by setting the alpha channel to 0, e.g.
// the expected input for a 4 channel image will be (0, 0, 0, 0) for pixels to
// be filled-in. The alpha channel is assumed to be the last channel.
// The filling is the result of a heat equation diffusion such that, after
// convergence, the value at a pixel to be filled in will be the average of its
// 4 neighbors. The solution is computed using a fast pyramidal approach which
// means the result is not exact for filling very small regions.
// The input and result images are allowed to be the same image.
// Example Usage:
// ByteImage img(1024, 300, 4);
// PopulateImage(&img);
// FillRegion(img, &img);
template <typename T, int NUM_CHANNELS>
void FillRegion(const cv::Mat_<cv::Vec<T,NUM_CHANNELS>>& input,
                cv::Mat_<cv::Vec<T,NUM_CHANNELS>>* result);

template <typename T, int NUM_CHANNELS>
void FillRegionNoAlloc(const cv::Mat_<cv::Vec<T,NUM_CHANNELS>>& input,
                       cv::Mat_<cv::Vec<T,NUM_CHANNELS>>* result);

// -------------------------- implementation ------------------------------

template <typename T, int NUM_CHANNELS>
void FillRegionNoAlloc(const cv::Mat_<cv::Vec<T,NUM_CHANNELS>>& input,
                       cv::Mat_<cv::Vec<T,NUM_CHANNELS>>* result) {
 /* CHECK_GT(input.Width(), 0);
  CHECK_GT(input.Height(), 0);*/

  // Calculate the number of levels of the pyramid.
  const int num_levels_width = ceil(logf(input.cols) / logf(2));
  const int num_levels_height = ceil(logf(input.rows) / logf(2));
  const int num_levels = std::max(num_levels_width, num_levels_height);

  // Create a pyramid of images, where each level is half the size of the
  // previous level.
  std::unique_ptr<cv::Mat_<cv::Vec<T,NUM_CHANNELS>>[]> pyramid(
      new cv::Mat_<cv::Vec<T,NUM_CHANNELS>>[num_levels]);
  for (int i = 0; i < num_levels; ++i) {
    if (i == 0) {
      BoxHalfSize(input, &pyramid[0]);
    } else {
      BoxHalfSize(pyramid[i - 1], &pyramid[i]);
    }
  }

  // Fill in the masked pixels (with alpha = 0) by upscaling the pyramid.
  for (int i = num_levels - 1; i > 0; --i) {
    BiLinearDoubleSizeWithMaskNoAlloc<T, NUM_CHANNELS>(
        pyramid[i], &pyramid[i - 1]);
  }
  //result->CopyFrom(input);
  input.copyTo(*result);
  BiLinearDoubleSizeWithMaskNoAlloc<T, NUM_CHANNELS>(pyramid[0],
                                                                    result);
  for(int i = 0; i < num_levels; ++i)
    pyramid[i].release();
}

template <typename T, int NUM_CHANNELS>
void FillRegion(const cv::Mat_<cv::Vec<T,NUM_CHANNELS>>& input,
                cv::Mat_<cv::Vec<T,NUM_CHANNELS>>* result) {
  assert(result != nullptr);
  //result->Allocate(input.Width(), input.Height());
  result->create(input.rows, input.cols);
  FillRegionNoAlloc(input, result);
}


#endif  // SAURON_STITCH_IMAGE_DIFFUSE_FILL_REGION_H_
