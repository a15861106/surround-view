#ifndef SAURON_STITCH_IMAGE_DIFFUSE_IMAGE_DIFFUSE_H_
#define SAURON_STITCH_IMAGE_DIFFUSE_IMAGE_DIFFUSE_H_

//#include "stitch/image_diffuse/wimage.h"
#include "fill_region.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>



template <typename MaskType, int NumMaskChannels, typename ImageType,
          int NumImageChannels>
bool DiffuseFromMaskedRegion(const cv::Mat_<cv::Vec<MaskType, NumMaskChannels>>& mask,
                             cv::Mat_<cv::Vec<ImageType, NumImageChannels>>* image) {

  cv::Mat_<cv::Vec<double, NumImageChannels + 1>> image_with_alpha(mask.rows, mask.cols);
  for(int row = 0; row < mask.rows; ++row)
  for(int col = 0; col < mask.cols; ++col)
  {
      for(int c = 0; c < NumImageChannels + 1; ++c)
        image_with_alpha(row, col)[c] = 0.0;
  }

  for (int row = 0; row < mask.rows; ++row) {
    double* out = (double *)image_with_alpha.ptr(row);
    const ImageType* in = (ImageType* )(image->ptr(row));
    const MaskType* mask_value = (MaskType* )mask.ptr(row);
    for (int col = 0; col < mask.cols; ++col) {
      if (*mask_value < 0) {
        // If the mask indicates we should fill this pixel then leave the
        // output as all zeros and increment our pointers to the next pixel.
        out += NumImageChannels + 1;
        in += NumImageChannels;
      } else {
        // If we are not filling this pixel then copy it from the source image
        // and set the alpha channel to 1.
        for (int channel = 0; channel < NumImageChannels; ++channel) {
          *out++ = *in++;
        }
        *out++ = 1.f;
      }
      mask_value += NumMaskChannels;
    }
  }
  
  FillRegionNoAlloc(image_with_alpha, &image_with_alpha);

  for (int row = 0; row < mask.rows; ++row) {
    const double* in = (double* )image_with_alpha.ptr(row);
    ImageType* out = (ImageType* )(image->ptr(row));
    for (int col = 0; col < mask.cols; ++col) {
      const double alpha = in[NumImageChannels];
      for (int channel = 0; channel < NumImageChannels; ++channel) {
        *out++ = static_cast<ImageType>(std::max(
            static_cast<double>(std::numeric_limits<ImageType>::min()),
            std::min(static_cast<double>(std::numeric_limits<ImageType>::max()),
                     *in++ / alpha)));
      }
      // Skip the alpha channel.
      in++;
    }
  }
  
  image_with_alpha.release();
  return true;
}

// Given an image and a mask, diffuse the input image into any regions that
// have a negative value in the first channel of the mask.
//
// This code is essentially a wrapper around vision/image/fill_region.h which
// deals with the necessary manipulations if want to keep your mask information
// in a separate image instead of in an extra channel in your input/output
// image.

template <typename MaskType, int NumMaskChannels, typename ImageType,
          int NumImageChannels>
bool EncapDiffuseFromMasked(cv::Mat_<cv::Vec<MaskType,NumMaskChannels>> mask,
cv::Mat_<cv::Vec<ImageType,NumImageChannels>> *image)
{
    DiffuseFromMaskedRegion(mask, image);
    return true;
}


#endif  // VR_RENDER_IMAGE_DIFFUSE_FROM_MASKED_REGION_H_
