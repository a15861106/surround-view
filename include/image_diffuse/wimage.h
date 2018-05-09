// WImage is a simple image class with the following goals:
//    1. All the data has explicit ownership to avoid memory leaks
//    2. No hidden allocations or copies for performance.
//    3. Can easily treat external data as an image
//    4. Easy to create images which are subsets of other images
//    5. Fast pixel access which can take advantage of number of channels
//          if known at compile time.
//    6. Easy access to OpenCV methods if desired.  If OpenCV cannot be used
//       for licensing reasons (e.g. on Android) or to avoid the overhead of
//       linking the large OpenCV library, then defining WIMAGE_NO_OPENCV
//       will provide the same functionality except a WImage cannot be
//       constructed from an IplImage and an IplImage cannot be obtained
//       from a .Ipl() method to pass to OpenCV functions.  Note that the
//       non-OpenCV version does not align rows to 4 byte boundaries like
//       the OpenCV version.
//
// The WImage class is the image class which provides the data accessors.
// The 'W' comes from the fact that it is also a wrapper around the popular
// but inconvenient IplImage class. A WImage can be constructed either using a
// WImageBuffer class which allocates and frees the data,
// or using a WImageView class which constructs a subimage or a view into
// external data.  The view class does no memory management.  Each class
// actually has two versions, one when the number of channels is known at
// compile time and one when it isn't.  Using the one with the number of
// channels specified can provide some compile time optimizations by using the
// fact that the number of channels is a constant.
//
// We use the convention (c,r) to refer to column c and row r with (0,0) being
// the upper left corner.  This is similar to standard Euclidean coordinates
// with the first coordinate varying in the horizontal direction and the second
// coordinate varying in the vertical direction.
// Thus (c,r) is usually in the domain [0, width) X [0, height)
//
// Example usage:
// WImageBuffer3_b  im(5, 7);  // Make a 5X7 3 channel image of type uint8
// WImageView3_b  sub_im(&im, 2, 2, 3, 3);  // 3X3 submatrix
// vector<uint8> vec(10, 3);
// WImageView1_b user_im(&vec.front(), 2, 5);  // 2X5 image w/ supplied data
//
// im.SetZero();  // same as cvSetZero(im.Ipl())
// *im(2, 3) = 15;  // Modify the element at column 2, row 3
// MySetRand(&sub_im);
// MySetRand(&user_im);
//
// // Copy the second row into the first.  This can be done with no memory
// // allocation and will use SSE if IPP is available.
// int w = im.Width();
// im.View(0, 0, w, 1).CopyFrom(im.View(0, 1, w, 1));
//
// // Doesn't care about source of data since using WImage
// void MySetRand(WImage_b* im) {  // Works with any number of channels
//   for (int r = 0; r < im->Height(); ++r) {
//     float* row = im->Row(r);
//     for (int c = 0; c < im->Width(); ++c) {
//        for (int ch = 0; ch < im->Channels(); ++ch, ++row) {
//          *row = static_cast<uint8>(random() % 256);
//        }
//     }
//   }
// }
//
// There is also access to multi-channel pixels as a single entity.  It
// only requires the data type to have an = operator.
// NOTE: only size is checked, but not actual types, so a float vector
// could be passed into an image of ints.
// Vector3_b blue1(32, 64, 96);
// WImageBuffer3_b img(5, 7);
// img.set(3, 5, blue1);
// const Vector3_b& pixel1 = img.get<Vector3_b>(4, 2);
//
// Channels containing non-basic types are also permitted even for
// multi-channel images but the sizeof the type must be 1, 2, 4, or 8
// bytes as these are restrictions on the size of objects that can be
// stored in an IplImage.
// struct MyStruct {
//   char a, b;
// };
// WImageBuffer<MyStruct, 3> img(5, 7);
// *img(3, 5) = {'a', 'b'};
//
// Functions that are not part of the basic image allocation, viewing, and
// access should come from OpenCV, except some useful functions that are not
// part of OpenCV can be found in wimage_util.h

#ifndef SAURON_STITCH_IMAGE_DIFFUSE_WIMAGE_H_
#define SAURON_STITCH_IMAGE_DIFFUSE_WIMAGE_H_

#include <algorithm>
//#include "stitch/image_diffuse/integral_types.h"
#include "base/type.h"
#include "stitch/image_diffuse/macros.h"

//#define WIMAGE_NO_OPENCV
#include "opencv2/opencv.hpp"
#include <gflags/gflags.h>
#include <glog/logging.h>

#ifndef WIMAGE_NO_OPENCV

#ifndef WIN32
// TODO(djfilip): Delete include of cv.h and cxcore.h after dependency fixes.
#include <opencv/cv.h>
#include <opencv/cxcore.h>
//#include <opencv/cxtypes.h>
#else
// Avoid soft links that are used in linux builds
// TODO(djfilip): Delete include of cv.h after dependency fixes.
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#endif
#endif

using sauron::int8;
using sauron::int16;
using sauron::int32;
using sauron::int64;
using sauron::uint8;
using sauron::uint16;
using sauron::uint32;
using sauron::uint64;

template <typename T> class WImage;  // IWYU pragma: keep
template <typename T> class WImageBuffer;
template <typename T> class WImageView;

template<typename T, int C> class WImageC;
template<typename T, int C> class WImageBufferC;
template<typename T, int C> class WImageViewC;
template<typename T, int C> class WImageSetC;

// Commonly used typedefs.
// Naming convention follows other geometric classes like Vector3_f.

// Unsigned 8-bit fixed-point.
typedef WImage<uint8>            WImage_b;
typedef WImageView<uint8>        WImageView_b;
typedef WImageBuffer<uint8>      WImageBuffer_b;

typedef WImageC<uint8, 1>        WImage1_b;
typedef WImageViewC<uint8, 1>    WImageView1_b;
typedef WImageBufferC<uint8, 1>  WImageBuffer1_b;
typedef WImageSetC<uint8, 1>     WImageSet1_b;

typedef WImageC<uint8, 3>        WImage3_b;
typedef WImageViewC<uint8, 3>    WImageView3_b;
typedef WImageBufferC<uint8, 3>  WImageBuffer3_b;
typedef WImageSetC<uint8, 3>     WImageSet3_b;

typedef WImageC<uint8, 4>        WImage4_b;
typedef WImageViewC<uint8, 4>    WImageView4_b;
typedef WImageBufferC<uint8, 4>  WImageBuffer4_b;
typedef WImageSetC<uint8, 4>     WImageSet4_b;

// Signed 32-bit floating-point.
typedef WImage<float>            WImage_f;
typedef WImageView<float>        WImageView_f;
typedef WImageBuffer<float>      WImageBuffer_f;

typedef WImageC<float, 1>        WImage1_f;
typedef WImageViewC<float, 1>    WImageView1_f;
typedef WImageBufferC<float, 1>  WImageBuffer1_f;
typedef WImageSetC<float, 1>     WImageSet1_f;

typedef WImageC<float, 2>        WImage2_f;
typedef WImageViewC<float, 2>    WImageView2_f;
typedef WImageBufferC<float, 2>  WImageBuffer2_f;
typedef WImageSetC<float, 2>     WImageSet2_f;

typedef WImageC<float, 3>        WImage3_f;
typedef WImageViewC<float, 3>    WImageView3_f;
typedef WImageBufferC<float, 3>  WImageBuffer3_f;
typedef WImageSetC<float, 3>     WImageSet3_f;

typedef WImageC<float, 4>        WImage4_f;
typedef WImageViewC<float, 4>    WImageView4_f;
typedef WImageBufferC<float, 4>  WImageBuffer4_f;
typedef WImageSetC<float, 4>     WImageSet4_f;

// There isn't a standard for signed and unsigned short so be more
// explicit in the typename for these cases.

// Signed 16-bit fixed-point.
typedef WImage<int16>            WImage_16s;
typedef WImageView<int16>        WImageView_16s;
typedef WImageBuffer<int16>      WImageBuffer_16s;

typedef WImageC<int16, 1>        WImage1_16s;
typedef WImageViewC<int16, 1>    WImageView1_16s;
typedef WImageBufferC<int16, 1>  WImageBuffer1_16s;
typedef WImageSetC<int16, 1>     WImageSet1_16s;

typedef WImageC<int16, 3>        WImage3_16s;
typedef WImageViewC<int16, 3>    WImageView3_16s;
typedef WImageBufferC<int16, 3>  WImageBuffer3_16s;
typedef WImageSetC<int16, 3>     WImageSet3_16s;

typedef WImageC<int16, 4>        WImage4_16s;
typedef WImageViewC<int16, 4>    WImageView4_16s;
typedef WImageBufferC<int16, 4>  WImageBuffer4_16s;
typedef WImageSetC<int16, 4>     WImageSet4_16s;

// Unsigned 16-bit fixed-point.
typedef WImage<uint16>           WImage_16u;
typedef WImageView<uint16>       WImageView_16u;
typedef WImageBuffer<uint16>     WImageBuffer_16u;

typedef WImageC<uint16, 1>       WImage1_16u;
typedef WImageViewC<uint16, 1>   WImageView1_16u;
typedef WImageBufferC<uint16, 1> WImageBuffer1_16u;
typedef WImageSetC<uint16, 1>    WImageSet1_16u;

typedef WImageC<uint16, 3>       WImage3_16u;
typedef WImageViewC<uint16, 3>   WImageView3_16u;
typedef WImageBufferC<uint16, 3> WImageBuffer3_16u;
typedef WImageSetC<uint16, 3>    WImageSet3_16u;

typedef WImageC<uint16, 4>       WImage4_16u;
typedef WImageViewC<uint16, 4>   WImageView4_16u;
typedef WImageBufferC<uint16, 4> WImageBuffer4_16u;
typedef WImageSetC<uint16, 4>    WImageSet4_16u;

// Define a WImageData when there is no OpenCV and otherwise it is an IplImage.
#ifdef WIMAGE_NO_OPENCV
struct WImageData {
  char* imageData;
  int width;
  int height;
  int nChannels;
  int widthStep;
  int depth;
};
#else
typedef IplImage WImageData;
#endif  // WIMAGE_NO_OPENCV

//
// WImage definitions
//
// This WImage class gives access to the data it refers to.  It can be
// constructed either by allocating the data with a WImageBuffer class or
// using the WImageView class to refer to a subimage or outside data.
template<typename T>
class WImage {
 public:
  typedef T BaseType;
  static const int kChannelSize = sizeof(T);

  // WImage is an abstract class with no other virtual methods so make the
  // destructor virtual.
  virtual ~WImage() = 0;

  // Accessors
  T* ImageData() { return reinterpret_cast<T*>(image_->imageData); }
  const T* ImageData() const {
    return reinterpret_cast<const T*>(image_->imageData);
  }

  // Dimensions
  int Width() const { return image_->width; }
  int Height() const { return image_->height; }

  // WidthStep is the number of bytes to go to the pixel with the next y coord
  int WidthStep() const { return image_->widthStep; }

  // Number of channels (not necessarily known at compile time for base class).
  int Channels() const { return image_->nChannels; }

  // Number of bytes per channel.
  int ChannelSize() const { return sizeof(T); }

  // Number of bytes per pixel.
  int PixelSize() const { return Channels() * ChannelSize(); }

  // Number of bits per channel, OR'ed with WIMAGE_DEPTH_SIGN for
  // signed integer types (per OpenCV convention).
  int Depth() const;

  // Accessors to a row
  inline T* Row(int r) {
    return reinterpret_cast<T*>(image_->imageData + r * WidthStep());
  }

  inline const T* Row(int r) const {
    return reinterpret_cast<const T*>(image_->imageData + r * WidthStep());
  }

  // Pixel accessors that return a pointer to the first channel
  // of the pixel at column c, row r
  inline T* operator() (int c, int r) { return Row(r) + c * Channels(); }

  inline const T* operator() (int c, int r) const {
    return Row(r) + c * Channels();
  }

  bool IsNull() const { return image_ == NULL; }

  // Copy the contents from another image.  Asserts that width and height are
  // the same.
  void CopyFrom(const WImage<T>& src);

  // Set contents to zero.
  void SetZero();

  // Set the contents of the image to the specified value. The parameter, value,
  // is a pointer to an array with as many elements as there are channels in the
  // image.
  void Set(const T *value);

  // Construct a view into a region of this image
  WImageView<T> View(int c, int r, int width, int height);

  // Construct a const view into a region of this const image.
  // This will trigger a type mismatch for invalid usage such as
  // void foo(const WImage1_f &im) {
  //   im.View(0,0,10,10).CopyFrom(...);
  // }
  // However, not entirely const-safe since one could assign the return value
  // to a non-const WImageView (see remark at WImageView::WImageView).
  const WImageView<T> View(int c, int r, int width, int height) const;

#ifndef WIMAGE_NO_OPENCV
  IplImage* Ipl() { return image_; }
  const IplImage* Ipl() const { return image_; }
#endif

 protected:
  friend class WImageView<T>;

  explicit WImage(WImageData* img) : image_(img) {
    if (img != NULL) {
     /* CHECK_EQ(Depth(), image_->depth);*/
    }
  }

  void SetData(WImageData* image) {
    if (image != NULL) {
     /* CHECK_EQ(Depth(), image->depth);*/
    }
    image_ = image;
  }

  WImageData* image_;

 private:
  DISALLOW_COPY_AND_ASSIGN(WImage);
};

//
// WImageC definition
//
// The base class for WImageViewC or WImageBufferC.  Like WImage except
// number of channels known at compile time.
template<typename T, int C>
class WImageC : public WImage<T> {
 public:
  typedef typename WImage<T>::BaseType BaseType;
  static const int kChannelSize = sizeof(T);
  static const int kChannels = C;
  static const int kPixelSize = C*sizeof(T);

  virtual ~WImageC() = 0;

  int Channels() const { return C; }

  // Construct a view into a region of this image
  WImageViewC<T, C> View(int c, int r, int width, int height);

  // Construct a const view into a region of this const image.
  // See remark at WImage::View() const.
  const WImageViewC<T, C> View(int c, int r, int width, int height) const;

  // Set a multi-channel pixel with a single call.  The only requirement is
  // that the input pixel value has = operator.
  // The contents of type V must match a C-element array of T and otherwise
  // the behavior is undefined.
  template<typename V>
  void set(int c, int r, const V& v) {
    assert(sizeof(V) == C*sizeof(T));
    *reinterpret_cast<V*>((*this)(c, r)) = v;
  }

  // Treat the pixel as a single unit when fetching.  This isn't typesafe when
  // the elements of V and the image pixels have the same length but are of
  // different types, e.g. integer and float, or signed and unsigned int.
  template<typename V>
  const V& get(int c, int r) const {
    assert(sizeof(V) == C*sizeof(T));
    return *reinterpret_cast<const V*>((*this)(c, r));
  }

 protected:
  friend class WImageViewC<T, C>;

  explicit WImageC(WImageData* image) : WImage<T>(image) {
    if (image != NULL) {
    /*  CHECK_EQ(image->nChannels, Channels());*/
    }
  }

  void SetData(WImageData* image) {
    if (image != NULL) {
     /* CHECK_EQ(image->nChannels, C);*/
    }
    WImage<T>::SetData(image);
  }

 private:
  // The derived View class can be copied so make this protected.
  DISALLOW_COPY_AND_ASSIGN(WImageC);
};

// Internal utility routines for managing WImage memory.
// TODO(skal): hide this class from public header.
class WImageDataUtil {
 public:
  // Initialize an image header without allocating the data.
  // Return false if dimensions are too large and result in overflow.
  static bool InitImageHeader(int width, int height, int nchannels,
                              int depth, WImageData* image);

  // Creates or resizes image data with the given dimensions and type,
  // storing the pointer in *image.
  static void Allocate(int width, int height, int nchannels, int depth,
                       WImageData** image);

  // Same as Allocate when using OpenCV which crashes when out of memory
  // because we cannot catch the thrown exception.
  // If WIMAGE_NO_OPENCV is defined then does not crash when out of memory and
  // instead sets the memory pointer is set to null which can be tested with
  // IsNull().
  static bool TryAllocate(int width, int height, int nchannels,
                          int depth, WImageData** image_ptr);

  // Destroys WImageData pointed to by *image and sets the pointer to null.
  static void ReleaseImage(WImageData** image);

  // Safe and checked computation of x*y.
  // If no overflow occurred, stores product in result and returns true.
  // Otherwise leaves result unchanged and returns false. *result can be passed
  // NULL, in which case the result is not stored.
  // NOTE: Deliberately avoiding SafeMultiply provided by
  // util/intops/safe_int_ops.h, to avoid adding non-PG3 dependencies to Android
  // code.
  static bool SafeMultiply(int x, int y, int* result) {
    // Method suggested by http://goto/safe_multiply
    typedef int64 Bigint;
    static_assert(sizeof(Bigint) >= 2*sizeof(int),
                  "Unable to detect overflow after multiplication");
    const Bigint big = static_cast<Bigint>(x) * static_cast<Bigint>(y);
    if (big > static_cast<Bigint>(INT_MIN) &&
        big < static_cast<Bigint>(INT_MAX)) {
      if (result != NULL) *result = static_cast<int>(big);
      return true;
    } else {
      return false;
    }
  }

  // Call to log when when a WImageBuffer takes ownership of an IplImage.
#ifndef WIMAGE_NO_OPENCV
  static void LogAdopt(IplImage* image);
#endif
};

//
// WImageBuffer definitions
//
// Image class which owns the data, so it can be allocated and is always
// freed.  It cannot be copied but can be explicitly cloned.
//
template<typename T>
class WImageBuffer : public WImage<T> {
 public:
  typedef typename WImage<T>::BaseType BaseType;

  WImageBuffer() : WImage<T>(NULL) {}
  virtual ~WImageBuffer() { ReleaseImage(); }

  WImageBuffer(int width, int height, int nchannels) : WImage<T>(NULL) {
    Allocate(width, height, nchannels);
  }

  // Allocate an image.  If the currently allocated memory is large enough to
  // accommodate the specified image, no memory reallocation is done, and the
  // routine returns quickly. Note that the memory size will not decrease, even
  // if a tiny image size is requested after an initial allocation of a huge
  // image. This is done in the interest of performance. If it is really desired
  // to minimize the allocated memory, call ReleaseImage() before calling
  // Allocate().  Crashes if there is not enough memory.
  void Allocate(int width, int height, int nchannels);

  // Same as Allocate when using OpenCV which crashes when out of memory
  // because we cannot catch the thrown exception.
  // If WIMAGE_NO_OPENCV is defined then does not crash when out of memory and
  // instead sets the memory pointer is set to null which can be tested with
  // IsNull().
  void TryAllocate(int width, int height, int nchannels);

  // Clone an image, which reallocates this image if of a different dimension.
  void CloneFrom(const WImage<T>& src) {
    Allocate(src.Width(), src.Height(), src.Channels());
    WImage<T>::CopyFrom(src);
  }

  // Release the image if it isn't null.
  void ReleaseImage() {
    WImageDataUtil::ReleaseImage(&image_);
  }

#ifndef WIMAGE_NO_OPENCV
  // Constructor which takes ownership of a given IplImage.
  explicit WImageBuffer(IplImage* img) : WImage<T>(img) {
    WImageDataUtil::LogAdopt(image_);
  }

  // Set the data to point to an image, releasing the old data.
  void SetIpl(IplImage* img) {
    ReleaseImage();
    WImage<T>::SetData(img);
  }
#endif

  // Swap two image buffers without consuming additional memory.
  // (It would make sense to also have a specialization of std::swap(a,b) for
  // WImageBuffer, but C++ won't allow: search for "partial specialization"
  // at the go/ link below.)
  void Swap(WImageBuffer* b) {
    using std::swap;  // go/using-std-swap
    swap(image_, b->image_);
  }

 protected:
  using WImage<T>::image_;

 private:
  // Avoid hidden allocations and copies.
  DISALLOW_COPY_AND_ASSIGN(WImageBuffer);
};

//
// WImageBufferC definition
//
// Image class that owns the data. Like WImageBuffer except number of channels
// known at compile time.
template<typename T, int C>
class WImageBufferC : public WImageC<T, C> {
 public:
  typedef typename WImage<T>::BaseType BaseType;

  WImageBufferC() : WImageC<T, C>(NULL) {}
  virtual ~WImageBufferC() { ReleaseImage(); }

  WImageBufferC(int width, int height) : WImageC<T, C>(NULL) {
    Allocate(width, height);
  }

  // Allocate an image.  If the currently allocated memory is large enough to
  // accommodate the specified image, no memory reallocation is done, and the
  // routine returns quickly.
  void Allocate(int width, int height);


  // Same as Allocate when using OpenCV which crashes when out of memory
  // because we cannot catch the thrown exception.
  // If WIMAGE_NO_OPENCV is defined then does not crash when out of memory and
  // instead sets the memory pointer is set to null which can be tested with
  // IsNull().
  void TryAllocate(int width, int height);

  // Clone an image, which reallocates this image if of a different dimension.
  void CloneFrom(const WImageC<T, C>& src) {
    Allocate(src.Width(), src.Height());
    WImageC<T, C>::CopyFrom(src);
  }

  // Release the image if it isn't null.
  void ReleaseImage() {
    WImageDataUtil::ReleaseImage(&image_);
  }

#ifndef WIMAGE_NO_OPENCV
  // Constructor which takes ownership of a given IplImage so releases
  // the image on destruction.
  explicit WImageBufferC(IplImage* img) : WImageC<T, C>(img) {
    WImageDataUtil::LogAdopt(image_);
  }

  // Set the data to point to an image, releasing the old data
  void SetIpl(IplImage* img) {
    ReleaseImage();
    WImageC<T, C>::SetData(img);
  }
#endif

  // Swap two image buffers without consuming additional memory.
  void Swap(WImageBufferC* b) {
    using std::swap;  // go/using-std-swap
    swap(image_, b->image_);
  }

 protected:
  using WImageC<T, C>::image_;

 private:
  DISALLOW_COPY_AND_ASSIGN(WImageBufferC);
};

//
// WImageView definitions
//
// View into an image class which allows treating a subimage as an image
// or treating external data as an image
//
template<typename T>
class WImageView : public WImage<T> {
 public:
  typedef typename WImage<T>::BaseType BaseType;

  WImageView();
  virtual ~WImageView() {}

  // Construct a subimage.  No checks are done that the subimage lies completely
  // inside the original image. The parameters (c, r) specify the upper left
  // corner of the subimage.
  WImageView(WImage<T>* img, int c, int r, int width, int height);

  // Refer to external data.
  // If not given, width_step is the smallest value that fits the data row
  // and is divisible by four. (four byte aligned).
  // If width_step is 0 then calculates width_step assuming no padding.
  WImageView(T* data, int width, int height, int channels, int width_step = -1);

  // Copy constructor which does a shallow copy to allow multiple views
  // of same data.
  // C++ unfortunately doesn't allow constructing a const object. So this
  // prevents us from having a const copy constructor and a non-const copy
  // constructor. The side effect is that a non-const view can be made from
  // a const image.
  WImageView(const WImageView<T>& img) : WImage<T>(NULL) {
    SetHeader(&img.header_);
  }

  // A view typically doesn't need to be implicitly constructed from a
  // WImageBuffer. Instead a simple WImage should be used since it can be upcast
  // from a View and a Buffer.
  explicit WImageView(const WImage<T>& img) : WImage<T>(NULL) {
    SetHeader(img.image_);
  }

  // Const-unsafe (see remark at WImageView::WImageView).
  WImageView& operator=(const WImage<T>& img) {
    SetHeader(img.image_);
    return *this;
  }
  WImageView& operator=(const WImageView<T>& img) {
    SetHeader(&img.header_);
    return *this;
  }

#ifndef WIMAGE_NO_OPENCV
  // Refer to external data.  This does NOT take ownership
  // of the supplied IplImage.
  explicit WImageView(IplImage* img) : WImage<T>(NULL) {
    SetHeader(img);
  }
#endif

 protected:
  WImageData header_;

 private:
  void SetHeader(const WImageData* img) {
    header_ = *img;
    WImage<T>::SetData(&header_);
  }
};

template<typename T, int C>
class WImageViewC : public WImageC<T, C> {
 public:
  typedef typename WImage<T>::BaseType BaseType;

  WImageViewC();
  virtual ~WImageViewC() {}

  // Construct a subimage.  Checks are done to confirm that the subimage lies
  // completely inside the original image.
  WImageViewC(WImageC<T, C>* img, int c, int r, int width, int height);

  // Refer to external data
  // If not given, width_step is the smallest value that fits the data row
  // and is divisible by four. (four byte aligned).
  // If width_step is 0 then calculates width_step assuming no padding.
  WImageViewC(T* data, int width, int height, int width_step = -1);

  // Copy constructor which does a shallow copy to allow multiple views
  // of same data.
  // Const-unsafe (see remark at WImageView::WImageView).
  WImageViewC(const WImageViewC<T, C>& img) : WImageC<T, C>(NULL) {
    SetHeader(&img.header_);
  }

  // A view typically doesn't need to be implicitly constructed from a
  // WImageBuffer. Instead a simple WImage should be used since it can be upcast
  // from a View and a Buffer.
  explicit WImageViewC(const WImageC<T, C>& img) : WImageC<T, C>(NULL) {
    SetHeader(img.image_);
  }

  // Const-unsafe (see remark at WImageView::WImageView).
  WImageViewC& operator=(const WImageC<T, C>& img) {
    SetHeader(img.image_);
    return *this;
  }
  WImageViewC& operator=(const WImageViewC<T, C>& img) {
    SetHeader(&img.header_);
    return *this;
  }

#ifndef WIMAGE_NO_OPENCV
  // Refer to external data.  This does NOT take ownership
  // of the supplied IplImage.
  explicit WImageViewC(IplImage* img) : WImageC<T, C>(img) {
    SetHeader(img);
  }
#endif

 protected:
  WImageData header_;

 private:
  void SetHeader(const WImageData* img) {
    header_ = *img;
    WImage<T>::SetData(&header_);
  }
};

#define WIMAGE_DEPTH_SIGN 0x80000000  // Same as IPL_DEPTH_SIGN from OpenCV.
template <typename T>
inline int WImage<T>::Depth() const {
  constexpr size_t sizeof_t = sizeof(T);
  assert(sizeof_t == 1 || sizeof_t == 2 || sizeof_t == 4
                 || sizeof_t == 8);
  return sizeof_t * 8;
}
template<>
inline int WImage<uint8>::Depth() const { return 8; }
template<>
inline int WImage<int8>::Depth() const { return (WIMAGE_DEPTH_SIGN | 8); }
template<>
inline int WImage<int16>::Depth() const { return (WIMAGE_DEPTH_SIGN | 16); }
template<>
inline int WImage<uint16>::Depth() const { return 16; }
template<>
inline int WImage<int32>::Depth() const { return (WIMAGE_DEPTH_SIGN | 32); }
template<>
inline int WImage<uint32>::Depth() const { return 32; }
template<>
inline int WImage<uint64>::Depth() const { return 64; }
template<>
inline int WImage<float>::Depth() const { return 32; }
template<>
inline int WImage<double>::Depth() const { return 64; }

//
// Pure virtual destructors still need to be defined.
//
template<typename T> inline WImage<T>::~WImage() {}
template<typename T, int C> inline WImageC<T, C>::~WImageC() {}

//
// WImageDataUtil Methods.
//
template<typename T>
inline void WImageBuffer<T>::Allocate(int width, int height, int nchannels) {
  WImageDataUtil::Allocate(width, height, nchannels, WImage<T>::Depth(),
                             &image_);
}

template<typename T, int C>
inline void WImageBufferC<T, C>::Allocate(int width, int height) {
  WImageDataUtil::Allocate(width, height, C, WImage<T>::Depth(), &image_);
}

template<typename T>
inline void WImageBuffer<T>::TryAllocate(int width, int height, int nchannels) {
  WImageDataUtil::TryAllocate(width, height, nchannels, WImage<T>::Depth(),
                             &image_);
}

template<typename T, int C>
inline void WImageBufferC<T, C>::TryAllocate(int width, int height) {
  WImageDataUtil::TryAllocate(width, height, C, WImage<T>::Depth(), &image_);
}

//
// WImage Methods.
//
template<typename T>
void WImage<T>::SetZero() {
  char* dptr = reinterpret_cast<char*>(ImageData());
  int row_bytes = Width() * PixelSize();
  for (int i = Height(); i > 0; --i) {
    memset(dptr, 0, row_bytes);
    dptr += WidthStep();
  }
}

template<typename T>
void WImage<T>::Set(const T* value) {
  for (int i = 0; i < Height(); ++i) {
    T* pixel_ptr = Row(i);
    for (int j = 0; j < Width(); ++j) {
      for (int c = 0; c < Channels(); ++c) {
        (*pixel_ptr++) = value[c];
      }
    }
  }
}

template<typename T>
void WImage<T>::CopyFrom(const WImage<T>& src) {
  // TODO(djfilip): handle case when src and dest image data is contiguous.
  //CHECK_EQ(Width(), src.Width());
  //CHECK_EQ(Height(), src.Height());
  //CHECK_EQ(Channels(), src.Channels());
  const char* src_row = reinterpret_cast<const char*>(src.ImageData());
  char* dst_row = reinterpret_cast<char*>(ImageData());
  int row_bytes = Width() * PixelSize();
  for (int i = Height(); i > 0; --i) {
    memcpy(dst_row, src_row, row_bytes);
    src_row += src.WidthStep();
    dst_row += WidthStep();
  }
}

//
// ImageView methods
//
template<typename T>
WImageView<T>::WImageView() : WImage<T>(NULL) {
  WImageDataUtil::InitImageHeader(0, 0, 0, WImage<T>::Depth(), &header_);
  header_.imageData = reinterpret_cast<char*>(NULL);
  WImage<T>::SetData(&header_);
}

template<typename T>
WImageView<T>::WImageView(WImage<T>* img,
                          int c, int r,
                          int width, int height)
    : WImage<T>(NULL) {
  assert(width >= 0 && height >= 0);
  assert(0 <= c);
  assert(0 <= r);
  assert(c + width <= img->Width());
  assert(r + height <= img->Height());

  header_ = *(img->image_);
  header_.imageData = reinterpret_cast<char*>((*img)(c, r));
  header_.width = width;
  header_.height = height;
  header_.nChannels = img->Channels();
  header_.widthStep = img->WidthStep();
  header_.depth = img->Depth();
  WImage<T>::SetData(&header_);
}

template<typename T>
WImageView<T>::WImageView(T* data, int width, int height,
                          int nchannels, int width_step)
    : WImage<T>(NULL) {
  assert(width >= 0 && height >= 0);
  assert(nchannels > 0);

  assert(WImageDataUtil::InitImageHeader(width, height, nchannels,
                                        WImage<T>::Depth(), &header_));
  header_.imageData = reinterpret_cast<char*>(data);
  if (width_step == 0) {
    header_.widthStep = width * nchannels * sizeof(T);
  } else if (width_step > 0) {
    header_.widthStep = width_step;
  }
  WImage<T>::SetData(&header_);
}

template<typename T, int C>
WImageViewC<T, C>::WImageViewC(WImageC<T, C>* img,
                               int c, int r,
                               int width, int height)
    : WImageC<T, C>(NULL) {
  assert(width >= 0 && height >= 0);
  assert(0 <= c);
  assert(0 <= r);
  assert(c + width <= img->Width());
  assert(r + height <= img->Height());

  header_ = *(img->image_);
  header_.imageData = reinterpret_cast<char*>((*img)(c, r));
  header_.width = width;
  header_.height = height;
  header_.nChannels = img->Channels();
  header_.widthStep = img->WidthStep();
  header_.depth = img->Depth();
  WImageC<T, C>::SetData(&header_);
}

template<typename T, int C>
WImageViewC<T, C>::WImageViewC() : WImageC<T, C>(NULL) {
  WImageDataUtil::InitImageHeader(0, 0, C, WImage<T>::Depth(), &header_);
  header_.imageData = reinterpret_cast<char*>(NULL);
  WImageC<T, C>::SetData(&header_);
}

template<typename T, int C>
WImageViewC<T, C>::WImageViewC(T* data, int width, int height,
                               int width_step)
    : WImageC<T, C>(NULL) {
  assert(width >= 0 && height >= 0);

  assert(WImageDataUtil::InitImageHeader(width, height, C, WImage<T>::Depth(),
                                        &header_));
  header_.imageData = reinterpret_cast<char*>(data);
  if (width_step == 0) {
    header_.widthStep = width * C * sizeof(T);
  } else if (width_step > 0) {
    header_.widthStep = width_step;
  }
  WImageC<T, C>::SetData(&header_);
}

// Construct a view into a region of an image
template<typename T>
WImageView<T> WImage<T>::View(int c, int r, int width, int height) {
  return WImageView<T>(this, c, r, width, height);
}

template<typename T>
const WImageView<T> WImage<T>::View(int c, int r, int width, int height) const {
  // Cast away constness since WImageView only has constructors for non-const
  // WImage.
  return WImageView<T>(const_cast<WImage<T>*>(this), c, r, width, height);
}

template<typename T, int C>
WImageViewC<T, C> WImageC<T, C>::View(int c, int r, int width, int height) {
  return WImageViewC<T, C>(this, c, r, width, height);
}

template<typename T, int C>
const WImageViewC<T, C> WImageC<T, C>::View(
    int c, int r, int width, int height) const {
  // Cast away constness since WImageViewC only has constructors for non-const
  // WImageC.
  return
      WImageViewC<T, C>(const_cast<WImageC<T, C>*>(this), c, r, width, height);
}

// The non-OpenCV versions of the WImageDataUtil members.
// These are defined here so that Android builds only need the .h file.
// They are defined for OpenCV in a .cc file so that logging can be enabled
// per google3 module.
//
// TODO(mgeorg) stevehsu said "consider moving this code to .cc file,
// updating the Android build rules."
#ifdef WIMAGE_NO_OPENCV
inline void WImageDataUtil::Allocate(int width, int height, int nchannels,
                                     int depth, WImageData** image_ptr) {
  // The Allocate method is defined to crash for OpenCV and non-OpenCV
  // if the memory cannot be allocated, so crash if TryAllocate fails in
  // the non-OpenCV version.
  CHECK(TryAllocate(width, height, nchannels, depth, image_ptr));
  CHECK((*image_ptr)->imageData);
}

inline bool WImageDataUtil::TryAllocate(int width, int height, int nchannels,
                                        int depth, WImageData** image_ptr) {
  if (*image_ptr != NULL) {
    WImageData* image = *image_ptr;
    int old_size, new_size;
    if (!SafeMultiply(image->widthStep, image->height, &old_size)) return false;
    char* const old_data = image->imageData;
    if (!InitImageHeader(width, height, nchannels, depth, image)) return false;
    CHECK(SafeMultiply(image->widthStep, image->height, &new_size));

    // Don't resize if there is enough memory.
    if (new_size <= old_size) {
      image->imageData = old_data;
    } else {
      delete[] old_data;
      image->imageData = new (std::nothrow) char[new_size];
    }
  } else {
    WImageData* image = new WImageData;
    if (!InitImageHeader(width, height, nchannels, depth, image)) {
      delete image;
      return false;  // overflow occurred
    }
    image->imageData =
        new (std::nothrow) char[image->widthStep * image->height];
    *image_ptr = image;
  }

  // Delete and set image_ptr to null if data couldn't be allocated.
  if (!(*image_ptr)->imageData) {
    delete *image_ptr;
    *image_ptr = NULL;
    return false;
  }

  return true;
}

inline void WImageDataUtil::ReleaseImage(WImageData** image_ptr) {
  WImageData* image = *image_ptr;
  if (image != NULL) {
    delete[] image->imageData;
    delete image;
    *image_ptr = NULL;
  }
}

inline bool WImageDataUtil::InitImageHeader(int width, int height,
                                            int nchannels, int depth,
                                            WImageData* header) {
  if (width < 0 || height < 0 || nchannels < 0) {
    LOG(ERROR) << "Negative size: "
               << "width: " << width << ", height: " << height
               << ", channels: " << nchannels;
    return false;
  }
  const int sizeofT = (depth & ~WIMAGE_DEPTH_SIGN) / 8;
  int stride;
  if (!SafeMultiply(width, nchannels * sizeofT, &stride)) {
    return false;
  }
  if (!SafeMultiply(stride, height, NULL)) {
    return false;
  }
  header->imageData = NULL;
  header->width = width;
  header->height = height;
  header->nChannels = nchannels;
  header->depth = depth;
  header->widthStep = stride;
  return true;
}
#else
// The OpenCV version of TryAllocate works differently than the non-OpenCV
// version.  See comments in TryAllocate declaration.
inline bool WImageDataUtil::TryAllocate(int width, int height, int nchannels,
                                        int depth, WImageData** image_ptr) {
  const int sizeofT = (depth & ~WIMAGE_DEPTH_SIGN) / 8;
  int stride;
  if (!SafeMultiply(width, nchannels * sizeofT, &stride)) {
    return false;
  }
  if (!SafeMultiply(stride, height, NULL)) {
    return false;
  }
  Allocate(width, height, nchannels, depth, image_ptr);
  return true;  // Allocate crashes if there is not enough memory.
}
#endif  // WIMAGE_NO_OPENCV

#endif  // IMAGE_WIMAGE_WIMAGE_H_
