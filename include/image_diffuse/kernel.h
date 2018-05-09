// Kernel definitions for image processing.

#ifndef SAURON_STITCH_IMAGE_DIFFUSE_KERNEL_H_
#define SAURON_STITCH_IMAGE_DIFFUSE_KERNEL_H_


// Box filter kernel. The general case for any arbitrary kernel size
// is not implemented.
template <typename T, int KERNEL_WIDTH, int KERNEL_HEIGHT>
struct BoxKernel;

// Kernel specialization for a 2x2 box filter.
template <typename T>
struct BoxKernel<T, 2, 2> {
  static const int kWidth = 2;
  static const int kHeight = 2;
  static const T kData[kWidth * kHeight];
  static const T kSum;
};
template <typename T>
const T BoxKernel<T, 2, 2>::kData[] = {1, 1,
                                       1, 1};
template <typename T>
const T BoxKernel<T, 2, 2>::kSum = 4;

// Gaussian kernel definition. The general case for any arbitrary kernel size
// is not implemented.
template <typename T, int KERNEL_WIDTH, int KERNEL_HEIGHT>
struct GaussianKernel;

// Kernel specialization for a 3x3 Gaussian.
template <typename T>
struct GaussianKernel<T, 3, 3> {
  static const int kWidth = 3;
  static const int kHeight = 3;
  static const T kData[kWidth * kHeight];
  static const T kSum;
};
template <typename T>
const T GaussianKernel<T, 3, 3>::kData[] = {1, 2, 1,
                                            2, 4, 2,
                                            1, 2, 1};
template <typename T>
const T GaussianKernel<T, 3, 3>::kSum = 16;

// Kernel specialization for a 1x3 Gaussian.
template <typename T>
struct GaussianKernel<T, 1, 3> {
  static const int kWidth = 1;
  static const int kHeight = 3;
  static const T kData[kWidth * kHeight];
  static const T kSum;
};
template <typename T>
const T GaussianKernel<T, 1, 3>::kData[] = {1, 2, 1};

template <typename T>
const T GaussianKernel<T, 1, 3>::kSum = 4;

// Kernel specialization for a 3x1 Gaussian.
template <typename T>
struct GaussianKernel<T, 3, 1> {
  static const int kWidth = 3;
  static const int kHeight = 1;
  static const T kData[kWidth * kHeight];
  static const T kSum;
};
template <typename T>
const T GaussianKernel<T, 3, 1>::kData[] = {1, 2, 1};

template <typename T>
const T GaussianKernel<T, 3, 1>::kSum = 4;

// Kernel specialization for a 4x4 Gaussian.
template <typename T>
struct GaussianKernel<T, 4, 4> {
  static const int kWidth = 4;
  static const int kHeight = 4;
  static const T kData[kWidth * kHeight];
  static const T kSum;
};
template <typename T>
const T GaussianKernel<T, 4, 4>::kData[] = {1, 3, 3, 1,
                                            3, 9, 9, 3,
                                            3, 9, 9, 3,
                                            1, 3, 3, 1};
template <typename T>
const T GaussianKernel<T, 4, 4>::kSum = 64;

// Kernel specialization for a 4x1 Gaussian.
template <typename T>
struct GaussianKernel<T, 4, 1> {
  static const int kWidth = 4;
  static const int kHeight = 1;
  static const T kData[kWidth * kHeight];
  static const T kSum;
};
template <typename T>
const T GaussianKernel<T, 4, 1>::kData[] = {1, 3, 3, 1};
template <typename T>
const T GaussianKernel<T, 4, 1>::kSum = 8;

// Kernel specialization for a 1x4 Gaussian.
template <typename T>
struct GaussianKernel<T, 1, 4> {
  static const int kWidth = 1;
  static const int kHeight = 4;
  static const T kData[kWidth * kHeight];
  static const T kSum;
};
template <typename T>
const T GaussianKernel<T, 1, 4>::kData[] = {1, 3, 3, 1};
template <typename T>
const T GaussianKernel<T, 1, 4>::kSum = 8;

// Kernel specialization for a 5x5 Gaussian.
// Note that the default Pascal pyramid kernel sums 256, so it does not work for
// T = unsigned char. We modify the kernel to sum 255 by removing 1 from
// the center pixel, making its weight 35.
template <typename T>
struct GaussianKernel<T, 5, 5> {
  static const int kWidth = 5;
  static const int kHeight = 5;
  static const T kData[kWidth * kHeight];
  static const T kSum;
};
template <typename T>
const T GaussianKernel<T, 5, 5>::kData[] = {1, 4, 6, 4, 1,
                                            4, 16, 24, 16, 4,
                                            6, 24, 35, 24, 6,
                                            4, 16, 24, 16, 4,
                                            1, 4, 6, 4, 1};
template <typename T>
const T GaussianKernel<T, 5, 5>::kSum = 255;

// Kernel specialization for a 5x1 Gaussian.
template <typename T>
struct GaussianKernel<T, 5, 1> {
  static const int kWidth = 5;
  static const int kHeight = 1;
  static const T kData[kWidth * kHeight];
  static const T kSum;
};
template <typename T>
const T GaussianKernel<T, 5, 1>::kData[] = {1, 4, 6, 4, 1};
template <typename T>
const T GaussianKernel<T, 5, 1>::kSum = 16;

// Kernel specialization for a 1x5 Gaussian.
template <typename T>
struct GaussianKernel<T, 1, 5> {
  static const int kWidth = 1;
  static const int kHeight = 5;
  static const T kData[kWidth * kHeight];
  static const T kSum;
};
template <typename T>
const T GaussianKernel<T, 1, 5>::kData[] = {1, 4, 6, 4, 1};
template <typename T>
const T GaussianKernel<T, 1, 5>::kSum = 16;

// Bi-linear kernel definitions. Used for image upsampling. There are
// four different 2x2 kernels inside a 3x3 window that are 90 degree rotations
// of each other. They rotate w.r.t. the center pixel in the 3x3 window.
template <typename T>
struct BiLinearTopLeftKernel {
  static const int kWidth = 2;
  static const int kHeight = 2;
  static const T kData[kWidth * kHeight];
  static const T kSum;
};
template <typename T>
const T BiLinearTopLeftKernel<T>::kData[] = {1, 3,
                                             3, 9};
template <typename T>
const T BiLinearTopLeftKernel<T>::kSum = 16;

template <typename T>
struct BiLinearTopRightKernel {
  static const int kWidth = 2;
  static const int kHeight = 2;
  static const T kData[kWidth * kHeight];
  static const T kSum;
};
template <typename T>
const T BiLinearTopRightKernel<T>::kData[] = {3, 1,
                                              9, 3};
template <typename T>
const T BiLinearTopRightKernel<T>::kSum = 16;

template <typename T>
struct BiLinearBottomLeftKernel {
  static const int kWidth = 2;
  static const int kHeight = 2;
  static const T kData[kWidth * kHeight];
  static const T kSum;
};
template <typename T>
const T BiLinearBottomLeftKernel<T>::kData[] = {3, 9,
                                                1, 3};
template <typename T>
const T BiLinearBottomLeftKernel<T>::kSum = 16;

template <typename T>
struct BiLinearBottomRightKernel {
  static const int kWidth = 2;
  static const int kHeight = 2;
  static const T kData[kWidth * kHeight];
  static const T kSum;
};
template <typename T>
const T BiLinearBottomRightKernel<T>::kData[] = {9, 3,
                                                 3, 1};
template <typename T>
const T BiLinearBottomRightKernel<T>::kSum = 16;

// Bi-quadratic kernel group. Used for image upsampling. The group
// defines four 2x2 kernels used in a 3x3 window:
// top-left, top-right, bottom-left, bottom-right.
// When this kernel is used to upsample levels in a pyramid, it approximates
// a bi-quadratic spline interpolation.
template <typename T>
struct BiLinearKernelGroup {
  static const int kWidth = 3;
  static const int kHeight = 3;
  typedef BiLinearTopLeftKernel<T> TopLeft;
  typedef BiLinearTopRightKernel<T> TopRight;
  typedef BiLinearBottomLeftKernel<T> BottomLeft;
  typedef BiLinearBottomRightKernel<T> BottomRight;
};


#endif  // SAURON_STITCH_IMAGE_DIFFUSE_KERNEL_H_
