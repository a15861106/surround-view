#ifndef SAURON_COLOR_TRANSFORM_COLOR_TRANSFORMER_H_
#define SAURON_COLOR_TRANSFORM_COLOR_TRANSFORMER_H_

#include <memory>

#include "eigen3/Eigen/Dense"
#include "opencv2/opencv.hpp"


// The mean and covariance of the color distribution in an image.
struct ImageStatistics {
  // Index of the image these statistics are for.
  int index = 0;

  // Mean values for image.
  double mean[3] = {0.0, 0.0, 0.0};

  // Covariance of image as a column major 3x3 matrix. It is expected
  // to be symmetric, positive semidefinite.
  double covariance[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

template <typename T>
struct ColorTransform {
  // y = A * x + b
  //
  // where data = [A b] is 3 x 4 column major matrix.
  T data[12];
};

struct ColorTransformOptions {
        // If true then the left image of the first pair corresponds to the right
        // image of the last pair.
  bool loop_is_closed = true;
        
  // Index of the image whose color properties all other images are transformed
  // to. If a negative index is given then we instead aim to transform to an
  // average of the images.
  int reference_image_id = -1;
        
  // Regularization weight that penalizes each transform for deviating from the
  // identity transform. It must be positive.
  double regularization = 0.05;
  
  // Bounds on the region of each image which is used for calculating
  // statistics as a fraction of the image width/height.
  double roi_top = 0.4;
  double roi_left = 0.4;
  double roi_width = 0.2;
  double roi_height = 0.2;
};

class ColorTransformer {
public:
  ColorTransformer(const std::vector<cv::Mat3f>& left_images,
                   const std::vector<cv::Mat3f>& right_images,
				   const std::vector<cv::Mat3f>& forward_flows,
				   const std::vector<cv::Mat3f>& backward_flows);
  
  ColorTransformer(const std::vector<cv::Mat>& images, 
                   const std::vector<cv::Mat1b>& masks);
    // Apply color correction to a set of image pairs which are structured as
    // follows:
    //  Given 2 sets of images(warpped) L_j and R_j for j = 0 to j = n - 1, the set of
    //  image pairs left_images (L) and right_images (R) are structured such that:
    //    1) L_j and R_j is already been warpped, so they observe the same scene
    //       content and so comparing image statistics between them is meaningfull.
    //       This only needs to hold for a specified rectangular region of the image.
    //    2) Optionally the initial images form a loop so that there are n pairs
    //       of transformed images.
    // We find a globally consistent set of color transforms between the original
    // images I_0 to I_(n-1) based on comparing statistics between each transformed
    // image pair L_j, R_j. These color transforms are then applied to each image,
    // converting these images to float precision so as not to lose information.
    //
    // The output vectors must already contain the correct number of output images.
  // Each output Mat should already has size match the inputs.
  void ColorCorrectImagePairLoop(const ColorTransformOptions& options,
                                 std::vector<cv::Mat3f>& left_output,
                                 std::vector<cv::Mat3f>& right_output);
  
  void ColorCorrectImagePairLoop(const ColorTransformOptions& options,
                                 std::vector<cv::Mat3f>& outputs);

private:
  void EstimateColorTransforms(const ColorTransformOptions& options);

  // Pre-multiply all transforms by the inverse of the average transform.
  //
  // Return value indicates if the computation of the centering transform was
  // successful or not. If it is false, then the values in transforms are not
  // modified.
  bool CenterColorTransformsOnAverage();

  // Given a set of pairs of image statistics, where each pair
  // represents the same scene content viewed by two images, find a set
  // of color transforms that map all images to have the same statistics
  // as the first image (or as close as is possible).
  //
  // This is done by minimizing the symmetrized Kullback-Lieblier
  // divergence between the color distributions of the images, given by
  // the vector constraints. It is assumed that the graph implied by the
  // pairwise constraints is connected.
  //
  // Upon successful return, transforms will contain num_images entries.
  //
  // reference_image_index is the index of the image whose color
  // distribution should be matched by all other images.
  //
  // Sometimes the minimization of just the Kullback-Leibler divergence
  // can be ill conditioned, in that case it is worth regularizing the
  // optimization by asking that the ColorTransforms not deviate from
  // the identity transform. The strenth of this regularization is
  // controlled by regularization_weight. It must be positive.
  bool ComputeConsistentColorTransforms(
    int num_images, int reference_image_index,
    const double regularization_weight,
    const std::vector<std::pair<ImageStatistics, ImageStatistics>>& constraints);

  std::vector<cv::Mat3f> left_images_;
  std::vector<cv::Mat3f> right_images_;
  std::vector<cv::Mat3f> forward_flows_;
  std::vector<cv::Mat3f> backward_flows_;
  std::vector<ColorTransform<double>> transforms_;

  std::vector<cv::Mat> images_;
  std::vector<cv::Mat1b> masks_;
};


#endif // SAURON_COLOR_TRANSFORM_COLOR_TRANSFORMER_H_
