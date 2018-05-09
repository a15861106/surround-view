#ifndef SAURON_COLOR_TRANSFORM_SYMMETRIZED_KULLBACK_LEIBLER_COST_H_
#define SAURON_COLOR_TRANSFORM_SYMMETRIZED_KULLBACK_LEIBLER_COST_H_

#include "ceres/ceres.h"


struct ImageStatistics;

// Cost functor to compute Symmetrized Kullback-Leibler divergence
// between a pair of three dimensional normal distributions as a
// function of a pair of affine transformations applied to each of
// them.
//
// This cost functor is numerically differentiated and used to match a
// set of color distributions using Ceres Solver.
class SymmetrizedKullbackLeiblerCost {
 public:
  // Symmetrized Kullback-Leibler divergence between a pair of
  // gaussian distributions (m1, S1) and (m2, S2) is
  //
  //  Tr(S_0^{-1}S_1) + Tr(S_1^{-1}S_0) +
  //  (m_0 - m_1)^T(S_0^{-1})(m_0 - m_1) +
  //  (m_0 - m_1)^T(S_1^{-1})(m_0 - m_1)
  //
  // Now suppose that (m_i, S_i) are obtained by affinely transforming
  // a distribution (mu_i, Sigma_i) by (A_i, b_i) then we have
  //
  // m_i = A_i * mu_i + b
  // S_i = A_i * Sigma_i * A_i^T
  //     = A_i * L_i * L_i^T * A_i^T
  //     = B_i * B_i^T
  //
  // where B = A * L.
  //
  // Which allows us to re-write the divergence above terms as a sum
  // of nonlinear squares in terms of (A_0, b_0), (A_1, b_1).
  //
  // Tr(S_0^{-1}S_1) = Tr((B_0B_0^T)^{-1}B_1B_1^T)
  //                 = Tr(B_0^{-T}B_0^{-1}B_1B_1^T)
  //                 = |B_0^{-1}B_1|^2_F
  //
  // Tr(S_1^{-1}S_0) = |B_1^{-1}B_0|^2_F
  //
  // (m_0 - m_1)^T(S_0^{-1})(m_0 - m_1) = |B_0^{-1}(m_0 - m_1)|^2
  //
  // (m_0 - m_1)^T(S_1^{-1})(m_0 - m_1) = |B_1^{-1}(m_0 - m_1)|^2
  //
  // If transform0 and transform1 contain the matrix [A_0, b_0] and
  // [A_1, b_1] in column major form respectively, then residuals on
  // return will contains
  //
  // residuals[0:9]   = vec(B_0^{-1}B_1)
  // residuals[9:18]  = vec(B_1^{-1}B_0)
  // residuals[18:21] = B_0^{-1}(m_0 - m_1)
  // residuals[21:24] = B_1^{-1}(m_0 - m_1)
  bool operator()(const double* transform0,
                  const double* transform1,
                  double* residuals) const;

  // Create a numerically differentiated CostFunction object using
  // this functor.
  static ceres::CostFunction* Create(const ImageStatistics& image0,
                                     const ImageStatistics& image1);

 private:
  // Private constructor to ensure that this class can only be
  // instantiated by calling the factory Create.
  SymmetrizedKullbackLeiblerCost() {}

  // Initialize the functor.
  //
  // image0 and image1 capture the mean and covariance of the color
  // distributions of a pair of images.
  //
  // The covariances must be symmetric positive definite
  // matrices. Only the lower triangular part of the matrix is
  // accessed.
  //
  // Returns false if either of the covariance matrices are rank
  // deficient and the Cholesky factorization failed.
  bool Init(const ImageStatistics& image0, const ImageStatistics& image1);

  double mu0_[3];
  double mu1_[3];
  double l0_[3 * 3];
  double l1_[3 * 3];
};

#endif  // SAURON_COLOR_TRANSFORM_SYMMETRIZED_KULLBACK_LEIBLER_COST_H_
