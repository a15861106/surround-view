// Unrolled convolution loops for small kernels. Current suported kernel sizes
// are 2x2, 3x3, 4x4 and 5x5.

#ifndef SAURON_STITCH_IMAGE_DIFFUSE_CONVOLUTION_LOOP_H_
#define SAURON_STITCH_IMAGE_DIFFUSE_CONVOLUTION_LOOP_H_


// Specializations of the convolution inner loop. The loop is unrolled for
// small kernel sizes.
template <typename PIXEL_TYPE, typename KERNEL, int NUM_CHANNELS,
          int KERNEL_WIDTH, int KERNEL_HEIGHT> struct InnerLoop;

// Convolution inner loop for 2x2 kernels.
template <typename PIXEL_TYPE, typename KERNEL, int NUM_CHANNELS>
struct InnerLoop<PIXEL_TYPE, KERNEL, NUM_CHANNELS, 2, 2> {
  static const int kWidth = KERNEL::kWidth;
  static const int kHeight = KERNEL::kHeight;
  static_assert(kWidth == 2, "Kernel must have the same size as the loop");
  static_assert(kHeight == 2, "Kernel must have the same size as the loop");

  static void Call(PIXEL_TYPE const* const row[], PIXEL_TYPE* res_ptr) {
    for (int z = 0; z < NUM_CHANNELS; ++z) {
      *(res_ptr++) =
          (KERNEL::kData[0] * row[0][z + NUM_CHANNELS * 0] +
           KERNEL::kData[1] * row[0][z + NUM_CHANNELS * 1] +
           KERNEL::kData[2] * row[1][z + NUM_CHANNELS * 0] +
           KERNEL::kData[3] * row[1][z + NUM_CHANNELS * 1]) / KERNEL::kSum;
    }
  }
};

// Convolution inner loop for 3x3 kernels.
template <typename PIXEL_TYPE, typename KERNEL, int NUM_CHANNELS>
struct InnerLoop<PIXEL_TYPE, KERNEL, NUM_CHANNELS, 3, 3> {
  static const int kWidth = KERNEL::kWidth;
  static const int kHeight = KERNEL::kHeight;
  static_assert(kWidth == 3, "Kernel must have the same size as the loop");
  static_assert(kHeight == 3, "Kernel must have the same size as the loop");

  static void Call(PIXEL_TYPE const* const row[], PIXEL_TYPE* res_ptr) {
    for (int z = 0; z < NUM_CHANNELS; ++z) {
      *(res_ptr++) =
          (KERNEL::kData[0] * row[0][z + NUM_CHANNELS * 0] +
           KERNEL::kData[1] * row[0][z + NUM_CHANNELS * 1] +
           KERNEL::kData[2] * row[0][z + NUM_CHANNELS * 2] +
           KERNEL::kData[3] * row[1][z + NUM_CHANNELS * 0] +
           KERNEL::kData[4] * row[1][z + NUM_CHANNELS * 1] +
           KERNEL::kData[5] * row[1][z + NUM_CHANNELS * 2] +
           KERNEL::kData[6] * row[2][z + NUM_CHANNELS * 0] +
           KERNEL::kData[7] * row[2][z + NUM_CHANNELS * 1] +
           KERNEL::kData[8] * row[2][z + NUM_CHANNELS * 2]) / KERNEL::kSum;
    }
  }
};

// Convolution inner loop for 3x1 kernels.
template <typename PIXEL_TYPE, typename KERNEL, int NUM_CHANNELS>
struct InnerLoop<PIXEL_TYPE, KERNEL, NUM_CHANNELS, 3, 1> {
  static const int kWidth = KERNEL::kWidth;
  static const int kHeight = KERNEL::kHeight;
  static_assert(kWidth == 3, "Kernel must have the same size as the loop");
  static_assert(kHeight == 1, "Kernel must have the same size as the loop");

  static void Call(PIXEL_TYPE const* const row[], PIXEL_TYPE* res_ptr) {
    for (int z = 0; z < NUM_CHANNELS; ++z) {
      *(res_ptr++) =
          (KERNEL::kData[0] * row[0][z + NUM_CHANNELS * 0] +
           KERNEL::kData[1] * row[0][z + NUM_CHANNELS * 1] +
           KERNEL::kData[2] * row[0][z + NUM_CHANNELS * 2]) / KERNEL::kSum;
    }
  }
};

// Convolution inner loop for 1x3 kernels.
template <typename PIXEL_TYPE, typename KERNEL, int NUM_CHANNELS>
struct InnerLoop<PIXEL_TYPE, KERNEL, NUM_CHANNELS, 1, 3> {
  static const int kWidth = KERNEL::kWidth;
  static const int kHeight = KERNEL::kHeight;
  static_assert(kWidth == 1, "Kernel must have the same size as the loop");
  static_assert(kHeight == 3, "Kernel must have the same size as the loop");

  static void Call(PIXEL_TYPE const* const row[], PIXEL_TYPE* res_ptr) {
    for (int z = 0; z < NUM_CHANNELS; ++z) {
      *(res_ptr++) =
          (KERNEL::kData[0] * row[0][z + NUM_CHANNELS * 0] +
           KERNEL::kData[1] * row[1][z + NUM_CHANNELS * 0] +
           KERNEL::kData[2] * row[2][z + NUM_CHANNELS * 0]) / KERNEL::kSum;
    }
  }
};

// Convolution inner loop for 4x4 kernels.
template <typename PIXEL_TYPE, typename KERNEL, int NUM_CHANNELS>
struct InnerLoop<PIXEL_TYPE, KERNEL, NUM_CHANNELS, 4, 4> {
  static const int kWidth = KERNEL::kWidth;
  static const int kHeight = KERNEL::kHeight;
  static_assert(kWidth == 4, "Kernel must have the same size as the loop");
  static_assert(kHeight == 4, "Kernel must have the same size as the loop");

  static void Call(PIXEL_TYPE const* const row[], PIXEL_TYPE* res_ptr) {
    for (int z = 0; z < NUM_CHANNELS; ++z) {
      *(res_ptr++) =
          (KERNEL::kData[0] * row[0][z + NUM_CHANNELS * 0] +
           KERNEL::kData[1] * row[0][z + NUM_CHANNELS * 1] +
           KERNEL::kData[2] * row[0][z + NUM_CHANNELS * 2] +
           KERNEL::kData[3] * row[0][z + NUM_CHANNELS * 3] +
           KERNEL::kData[4] * row[1][z + NUM_CHANNELS * 0] +
           KERNEL::kData[5] * row[1][z + NUM_CHANNELS * 1] +
           KERNEL::kData[6] * row[1][z + NUM_CHANNELS * 2] +
           KERNEL::kData[7] * row[1][z + NUM_CHANNELS * 3] +
           KERNEL::kData[8] * row[2][z + NUM_CHANNELS * 0] +
           KERNEL::kData[9] * row[2][z + NUM_CHANNELS * 1] +
           KERNEL::kData[10] * row[2][z + NUM_CHANNELS * 2] +
           KERNEL::kData[11] * row[2][z + NUM_CHANNELS * 3] +
           KERNEL::kData[12] * row[3][z + NUM_CHANNELS * 0] +
           KERNEL::kData[13] * row[3][z + NUM_CHANNELS * 1] +
           KERNEL::kData[14] * row[3][z + NUM_CHANNELS * 2] +
           KERNEL::kData[15] * row[3][z + NUM_CHANNELS * 3]) / KERNEL::kSum;
    }
  }
};

// Convolution inner loop for 4x1 kernels.
template <typename PIXEL_TYPE, typename KERNEL, int NUM_CHANNELS>
struct InnerLoop<PIXEL_TYPE, KERNEL, NUM_CHANNELS, 4, 1> {
  static const int kWidth = KERNEL::kWidth;
  static const int kHeight = KERNEL::kHeight;
  static_assert(kWidth == 4, "Kernel must have the same size as the loop");
  static_assert(kHeight == 1, "Kernel must have the same size as the loop");

  static void Call(PIXEL_TYPE const* const row[], PIXEL_TYPE* res_ptr) {
    for (int z = 0; z < NUM_CHANNELS; ++z) {
      *(res_ptr++) =
          (KERNEL::kData[0] * row[0][z + NUM_CHANNELS * 0] +
           KERNEL::kData[1] * row[0][z + NUM_CHANNELS * 1] +
           KERNEL::kData[2] * row[0][z + NUM_CHANNELS * 2] +
           KERNEL::kData[3] * row[0][z + NUM_CHANNELS * 3]) / KERNEL::kSum;
    }
  }
};

// Convolution inner loop for 1x4 kernels.
template <typename PIXEL_TYPE, typename KERNEL, int NUM_CHANNELS>
struct InnerLoop<PIXEL_TYPE, KERNEL, NUM_CHANNELS, 1, 4> {
  static const int kWidth = KERNEL::kWidth;
  static const int kHeight = KERNEL::kHeight;
  static_assert(kWidth == 1, "Kernel must have the same size as the loop");
  static_assert(kHeight == 4, "Kernel must have the same size as the loop");

  static void Call(PIXEL_TYPE const* const row[], PIXEL_TYPE* res_ptr) {
    for (int z = 0; z < NUM_CHANNELS; ++z) {
      *(res_ptr++) =
          (KERNEL::kData[0] * row[0][z] +
           KERNEL::kData[1] * row[1][z] +
           KERNEL::kData[2] * row[2][z] +
           KERNEL::kData[3] * row[3][z]) / KERNEL::kSum;
    }
  }
};

// Convolution inner loop for 5x5 kernels.
template <typename PIXEL_TYPE, typename KERNEL, int NUM_CHANNELS>
struct InnerLoop<PIXEL_TYPE, KERNEL, NUM_CHANNELS, 5, 5> {
  static const int kWidth = KERNEL::kWidth;
  static const int kHeight = KERNEL::kHeight;
  static_assert(kWidth == 5, "Kernel must have the same size as the loop");
  static_assert(kHeight == 5, "Kernel must have the same size as the loop");

  static void Call(PIXEL_TYPE const* const row[], PIXEL_TYPE* res_ptr) {
    for (int z = 0; z < NUM_CHANNELS; ++z) {
      *(res_ptr++) =
          (KERNEL::kData[0] * row[0][z + NUM_CHANNELS * 0] +
           KERNEL::kData[1] * row[0][z + NUM_CHANNELS * 1] +
           KERNEL::kData[2] * row[0][z + NUM_CHANNELS * 2] +
           KERNEL::kData[3] * row[0][z + NUM_CHANNELS * 3] +
           KERNEL::kData[4] * row[0][z + NUM_CHANNELS * 4] +
           KERNEL::kData[5] * row[1][z + NUM_CHANNELS * 0] +
           KERNEL::kData[6] * row[1][z + NUM_CHANNELS * 1] +
           KERNEL::kData[7] * row[1][z + NUM_CHANNELS * 2] +
           KERNEL::kData[8] * row[1][z + NUM_CHANNELS * 3] +
           KERNEL::kData[9] * row[1][z + NUM_CHANNELS * 4] +
           KERNEL::kData[10] * row[2][z + NUM_CHANNELS * 0] +
           KERNEL::kData[11] * row[2][z + NUM_CHANNELS * 1] +
           KERNEL::kData[12] * row[2][z + NUM_CHANNELS * 2] +
           KERNEL::kData[13] * row[2][z + NUM_CHANNELS * 3] +
           KERNEL::kData[14] * row[2][z + NUM_CHANNELS * 4] +
           KERNEL::kData[15] * row[3][z + NUM_CHANNELS * 0] +
           KERNEL::kData[16] * row[3][z + NUM_CHANNELS * 1] +
           KERNEL::kData[17] * row[3][z + NUM_CHANNELS * 2] +
           KERNEL::kData[18] * row[3][z + NUM_CHANNELS * 3] +
           KERNEL::kData[19] * row[3][z + NUM_CHANNELS * 4] +
           KERNEL::kData[20] * row[4][z + NUM_CHANNELS * 0] +
           KERNEL::kData[21] * row[4][z + NUM_CHANNELS * 1] +
           KERNEL::kData[22] * row[4][z + NUM_CHANNELS * 2] +
           KERNEL::kData[23] * row[4][z + NUM_CHANNELS * 3] +
           KERNEL::kData[24] * row[4][z + NUM_CHANNELS * 4]) / KERNEL::kSum;
    }
  }
};

// Convolution inner loop for 5x1 kernels.
template <typename PIXEL_TYPE, typename KERNEL, int NUM_CHANNELS>
struct InnerLoop<PIXEL_TYPE, KERNEL, NUM_CHANNELS, 5, 1> {
  static const int kWidth = KERNEL::kWidth;
  static const int kHeight = KERNEL::kHeight;
  static_assert(kWidth == 5, "Kernel must have the same size as the loop");
  static_assert(kHeight == 1, "Kernel must have the same size as the loop");

  static void Call(PIXEL_TYPE const* const row[], PIXEL_TYPE* res_ptr) {
    for (int z = 0; z < NUM_CHANNELS; ++z) {
      *(res_ptr++) =
          (KERNEL::kData[0] * row[0][z + NUM_CHANNELS * 0] +
           KERNEL::kData[1] * row[0][z + NUM_CHANNELS * 1] +
           KERNEL::kData[2] * row[0][z + NUM_CHANNELS * 2] +
           KERNEL::kData[3] * row[0][z + NUM_CHANNELS * 3] +
           KERNEL::kData[4] * row[0][z + NUM_CHANNELS * 4]) / KERNEL::kSum;
    }
  }
};

// Convolution inner loop for 1x5 kernels.
template <typename PIXEL_TYPE, typename KERNEL, int NUM_CHANNELS>
struct InnerLoop<PIXEL_TYPE, KERNEL, NUM_CHANNELS, 1, 5> {
  static const int kWidth = KERNEL::kWidth;
  static const int kHeight = KERNEL::kHeight;
  static_assert(kWidth == 1, "Kernel must have the same size as the loop");
  static_assert(kHeight == 5, "Kernel must have the same size as the loop");

  static void Call(PIXEL_TYPE const* const row[], PIXEL_TYPE* res_ptr) {
    for (int z = 0; z < NUM_CHANNELS; ++z) {
      *(res_ptr++) =
          (KERNEL::kData[0] * row[0][z] +
           KERNEL::kData[1] * row[1][z] +
           KERNEL::kData[2] * row[2][z] +
           KERNEL::kData[3] * row[3][z] +
           KERNEL::kData[4] * row[4][z]) / KERNEL::kSum;
    }
  }
};

// An inner loop that uses a kernel group instead of a single kernel. This loop
// is used when different pixels use different kernels. A prominent use case
// is upsampling an image where a pixel uses a kernel depending on whether its
// row and column is odd or even. E.g. for the case of 3x3 kernels, a common
// group is composed of four 2x2 kernels, noted:
// top-left, top-right, bottom-left, bottom-right.

// As an example, given the following 3x3 window:
//  A   B   C
//  D   E   F
//  G   H   I
//
// We have the following group of four 2x2 kernels:
// top-left   top-right    bottom-left   bottom-right
//  A   B       B   C         D   E         E   F
//  D   E       E   F         G   H         H   I
template <typename PIXEL_TYPE, typename KERNEL_GROUP, int NUM_CHANNELS,
          int KERNEL_WIDTH, int KERNEL_HEIGHT>
struct InnerLoopWithGroup;

// InnerLoopWithGroup for 3x3 kernels.
template <typename PIXEL_TYPE, typename KERNEL_GROUP, int NUM_CHANNELS>
struct InnerLoopWithGroup<PIXEL_TYPE, KERNEL_GROUP, NUM_CHANNELS, 3, 3> {
  static const int kWidth = KERNEL_GROUP::kWidth;
  static const int kHeight = KERNEL_GROUP::kHeight;
  static_assert(kWidth == 3, "Kernel must have the same size as the loop");
  static_assert(kHeight == 3, "Kernel must have the same size as the loop");

  static void TopLeft(PIXEL_TYPE const* const row[], PIXEL_TYPE* res_ptr) {
    for (int z = 0; z < NUM_CHANNELS; ++z) {
      *(res_ptr++) =
          (KERNEL_GROUP::TopLeft::kData[0] * row[0][z + NUM_CHANNELS * 0] +
           KERNEL_GROUP::TopLeft::kData[1] * row[0][z + NUM_CHANNELS * 1] +
           KERNEL_GROUP::TopLeft::kData[2] * row[1][z + NUM_CHANNELS * 0] +
           KERNEL_GROUP::TopLeft::kData[3] * row[1][z + NUM_CHANNELS * 1]) /
          KERNEL_GROUP::TopLeft::kSum;
    }
  }

  static void TopRight(PIXEL_TYPE const* const row[], PIXEL_TYPE* res_ptr) {
    for (int z = 0; z < NUM_CHANNELS; ++z) {
      *(res_ptr++) =
          (KERNEL_GROUP::TopRight::kData[0] * row[0][z + NUM_CHANNELS * 1] +
           KERNEL_GROUP::TopRight::kData[1] * row[0][z + NUM_CHANNELS * 2] +
           KERNEL_GROUP::TopRight::kData[2] * row[1][z + NUM_CHANNELS * 1] +
           KERNEL_GROUP::TopRight::kData[3] * row[1][z + NUM_CHANNELS * 2]) /
          KERNEL_GROUP::TopRight::kSum;
    }
  }

  static void BottomLeft(PIXEL_TYPE const* const row[], PIXEL_TYPE* res_ptr) {
    for (int z = 0; z < NUM_CHANNELS; ++z) {
      *(res_ptr++) =
          (KERNEL_GROUP::BottomLeft::kData[0] * row[1][z + NUM_CHANNELS * 0] +
           KERNEL_GROUP::BottomLeft::kData[1] * row[1][z + NUM_CHANNELS * 1] +
           KERNEL_GROUP::BottomLeft::kData[2] * row[2][z + NUM_CHANNELS * 0] +
           KERNEL_GROUP::BottomLeft::kData[3] * row[2][z + NUM_CHANNELS * 1]) /
          KERNEL_GROUP::BottomLeft::kSum;
    }
  }

  static void BottomRight(PIXEL_TYPE const* const row[], PIXEL_TYPE* res_ptr) {
    for (int z = 0; z < NUM_CHANNELS; ++z) {
      *(res_ptr++) =
          (KERNEL_GROUP::BottomRight::kData[0] * row[1][z + NUM_CHANNELS * 1] +
           KERNEL_GROUP::BottomRight::kData[1] * row[1][z + NUM_CHANNELS * 2] +
           KERNEL_GROUP::BottomRight::kData[2] * row[2][z + NUM_CHANNELS * 1] +
           KERNEL_GROUP::BottomRight::kData[3] * row[2][z + NUM_CHANNELS * 2])
          / KERNEL_GROUP::BottomRight::kSum;
    }
  }

  // This function outputs two consecutive samples.
  static void Call(PIXEL_TYPE const* const rows[],
                   bool row_is_odd,
                   PIXEL_TYPE* res_ptr) {
    if (row_is_odd) {
      BottomLeft(rows, res_ptr);
      res_ptr += NUM_CHANNELS;
      BottomRight(rows, res_ptr);
    } else {
      TopLeft(rows, res_ptr);
      res_ptr += NUM_CHANNELS;
      TopRight(rows, res_ptr);
    }
  }

  // This function outputs one or two consecutive samples.
  static void Call(PIXEL_TYPE const* const rows[],
                   bool row_is_odd,
                   bool output_two_samples,
                   PIXEL_TYPE* res_ptr) {
    if (row_is_odd) {
      BottomLeft(rows, res_ptr);
      if (output_two_samples) {
        res_ptr += NUM_CHANNELS;
        BottomRight(rows, res_ptr);
      }
    } else {
      TopLeft(rows, res_ptr);
      if (output_two_samples) {
        res_ptr += NUM_CHANNELS;
        TopRight(rows, res_ptr);
      }
    }
  }
};

// An InnerLoopGroup that treats the last channel as
// an alpha channel, and only perform the write on pixels whose alpha channel
// is 0.
template <typename PIXEL_TYPE, typename KERNEL_GROUP, int NUM_CHANNELS,
          int KERNEL_WIDTH, int KERNEL_HEIGHT>
struct MaskedInnerLoopWithGroup;

// InnerLoopGroupWithMask for 3x3 kernels.
template <typename PIXEL_TYPE, typename KERNEL_GROUP, int NUM_CHANNELS>
struct MaskedInnerLoopWithGroup<PIXEL_TYPE, KERNEL_GROUP, NUM_CHANNELS, 3, 3> {
  static const int kWidth = KERNEL_GROUP::kWidth;
  static const int kHeight = KERNEL_GROUP::kHeight;
  static_assert(kWidth == 3, "Kernel must have the same size as the loop");
  static_assert(kHeight == 3, "Kernel must have the same size as the loop");

  static void TopLeft(PIXEL_TYPE const* const row[], PIXEL_TYPE* res_ptr) {
    if (res_ptr[NUM_CHANNELS - 1] == PIXEL_TYPE(0)) {
      for (int z = 0; z < NUM_CHANNELS; ++z) {
        *(res_ptr++) =
          (KERNEL_GROUP::TopLeft::kData[0] * row[0][z + NUM_CHANNELS * 0] +
           KERNEL_GROUP::TopLeft::kData[1] * row[0][z + NUM_CHANNELS * 1] +
           KERNEL_GROUP::TopLeft::kData[2] * row[1][z + NUM_CHANNELS * 0] +
           KERNEL_GROUP::TopLeft::kData[3] * row[1][z + NUM_CHANNELS * 1]) /
          KERNEL_GROUP::TopLeft::kSum;
      }
    }
  }

  static void TopRight(PIXEL_TYPE const* const row[], PIXEL_TYPE* res_ptr) {
    if (res_ptr[NUM_CHANNELS - 1] == PIXEL_TYPE(0)) {
      for (int z = 0; z < NUM_CHANNELS; ++z) {
        *(res_ptr++) =
          (KERNEL_GROUP::TopRight::kData[0] * row[0][z + NUM_CHANNELS * 1] +
           KERNEL_GROUP::TopRight::kData[1] * row[0][z + NUM_CHANNELS * 2] +
           KERNEL_GROUP::TopRight::kData[2] * row[1][z + NUM_CHANNELS * 1] +
           KERNEL_GROUP::TopRight::kData[3] * row[1][z + NUM_CHANNELS * 2]) /
          KERNEL_GROUP::TopRight::kSum;
      }
    }
  }

  static void BottomLeft(PIXEL_TYPE const* const row[], PIXEL_TYPE* res_ptr) {
    if (res_ptr[NUM_CHANNELS - 1] == PIXEL_TYPE(0)) {
      for (int z = 0; z < NUM_CHANNELS; ++z) {
        *(res_ptr++) =
          (KERNEL_GROUP::BottomLeft::kData[0] * row[1][z + NUM_CHANNELS * 0] +
           KERNEL_GROUP::BottomLeft::kData[1] * row[1][z + NUM_CHANNELS * 1] +
           KERNEL_GROUP::BottomLeft::kData[2] * row[2][z + NUM_CHANNELS * 0] +
           KERNEL_GROUP::BottomLeft::kData[3] * row[2][z + NUM_CHANNELS * 1]) /
          KERNEL_GROUP::BottomLeft::kSum;
      }
    }
  }

  static void BottomRight(PIXEL_TYPE const* const row[], PIXEL_TYPE* res_ptr) {
    if (res_ptr[NUM_CHANNELS - 1] == PIXEL_TYPE(0)) {
      for (int z = 0; z < NUM_CHANNELS; ++z) {
        *(res_ptr++) =
          (KERNEL_GROUP::BottomRight::kData[0] * row[1][z + NUM_CHANNELS * 1] +
           KERNEL_GROUP::BottomRight::kData[1] * row[1][z + NUM_CHANNELS * 2] +
           KERNEL_GROUP::BottomRight::kData[2] * row[2][z + NUM_CHANNELS * 1] +
           KERNEL_GROUP::BottomRight::kData[3] * row[2][z + NUM_CHANNELS * 2])
            / KERNEL_GROUP::BottomRight::kSum;
      }
    }
  }

  // This function outputs two consecutive samples.
  static void Call(PIXEL_TYPE const* const rows[],
                   bool row_is_odd,
                   PIXEL_TYPE* res_ptr) {
    if (row_is_odd) {
      BottomLeft(rows, res_ptr);
      res_ptr += NUM_CHANNELS;
      BottomRight(rows, res_ptr);
    } else {
      TopLeft(rows, res_ptr);
      res_ptr += NUM_CHANNELS;
      TopRight(rows, res_ptr);
    }
  }

  // This function outputs one or two consecutive samples.
  static void Call(PIXEL_TYPE const* const rows[],
                   bool row_is_odd,
                   bool output_two_samples,
                   PIXEL_TYPE* res_ptr) {
    if (row_is_odd) {
      BottomLeft(rows, res_ptr);
      if (output_two_samples) {
        res_ptr += NUM_CHANNELS;
        BottomRight(rows, res_ptr);
      }
    } else {
      TopLeft(rows, res_ptr);
      if (output_two_samples) {
        res_ptr += NUM_CHANNELS;
        TopRight(rows, res_ptr);
      }
    }
  }
};

#endif  // VISION_IMAGE_CONVOLUTION_LOOP_H_

