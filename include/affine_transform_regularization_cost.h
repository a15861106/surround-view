#ifndef SAURON_COLOR_TRANSFORM_IMAGE_AFFINE_TRANSFORM_REGULARIZATION_COST_H_
#define SAURON_COLOR_TRANSFORM_IMAGE_AFFINE_TRANSFORM_REGULARIZATION_COST_H_

#include "ceres/ceres.h"


// An automatically differentiated Ceres Solver cost function to
// regularize an affine tranform (in column major form) by computing
// the Frobenius norm of its difference from an identity
// transformation.
class AffineTransformRegularizationCost {
 public:
  template <typename T>
  bool operator()(const T* affine_transform, T* residuals) const {
    for (int i = 0; i < 12; ++i) {
      residuals[i] = affine_transform[i];
    }

    residuals[0] = residuals[0] - T(1.0);
    residuals[4] = residuals[4] - T(1.0);
    residuals[8] = residuals[8] - T(1.0);

    return true;
  }

  static ceres::CostFunction* Create() {
    return new ceres::AutoDiffCostFunction<AffineTransformRegularizationCost,
                                           12, 12>(
        new AffineTransformRegularizationCost);
  }
};

#endif  // SAURON_COLOR_TRANSFORM_IMAGE_AFFINE_TRANSFORM_REGULARIZATION_COST_H_
