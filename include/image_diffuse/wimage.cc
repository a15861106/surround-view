// Utility routines for managing IplImage memory
//
// Set the flag "--vmodule wimage=1" to profile image memory usage.
//
// A log message of the form
//   8871 WImageBuffer::Allocate 2540*3142*3*1 = 23942040
// provides a serial number 8871, the image size 2540*3142, channels 3,
// bytes per pixel 1, and image bytes 23942040 (ignoring padding and other
// overhead).
//
// To find out what is causing a large allocation, run your code in gdb
// and set a breakpoint after datasize is computed in Allocate(), e.g.
//   break wimage.cc:53 if datasize>=20000000
//
// To investigate a specific allocation identified by serial number, set
// a breakpoint after serial is computed in Serial(), e.g.
//   break wimage.cc:41 if serial==8871
#include "stitch/image_diffuse/wimage.h"

#include <stddef.h>
#include <iostream>

//namespace {
//// Thread-safe generator of unique serial numbers, useful for logging.
//int Serial() {
//  // Wrapping the call to GetNext() rather than calling it directly from
//  // other WImageDataUtil methods facilitates setting a conditional breakpoint
//  // on the value of serial.
// // static base::SequenceNumber generator;
//  //int serial = generator.GetNext();
//  return serial;
//}
//}  // namespace
#ifndef WIMAGE_NO_OPENCV
void WImageDataUtil::Allocate(int width, int height, int nchannels, int depth,
                              WImageData** image) {
  /*CHECK(width >= 0 && height >= 0) << "Negative size: "
    << "width: " << width << ", height: " << height
    << ", channels: " << nchannels;*/
  const int sizeofT = (depth & ~IPL_DEPTH_SIGN) / 8;
  int stride;
  int datasize;
  //CHECK(SafeMultiply(width, nchannels * sizeofT, &stride));  // ignoring padding
  //CHECK(SafeMultiply(stride, height, &datasize));

  IplImage *ipl = *image;
  if (ipl != NULL) {
  /*  CHECK_EQ(depth, ipl->depth) << "Mismatched ipl->depth";*/
    size_t min_width_step = (stride + ipl->align - 1) & (~(ipl->align - 1));
    size_t needed_bytes = min_width_step * height;
    if (ipl->imageSize >= needed_bytes) {
      // Reuse existing buffer for a smaller image.
      // No bytes are actually being freed, but in order for image alloc/dealloc
      // logging to not seem to leak memory, log as if dealloc then realloc.
      const size_t old_datasize = ipl->width*ipl->height*ipl->nChannels*sizeofT;
     /* VLOG(1) << Serial() << " WImageBuffer::Allocate "
              << ipl->width << "*" << ipl->height
              << "*" << ipl->nChannels << "*" << sizeofT
              << " = -" << old_datasize;
      VLOG(1) << Serial() << " WImageBuffer::Allocate "
              << width << "*" << height << "*" << nchannels << "*" << sizeofT
              << " = " << datasize;*/
      ipl->width  = width;
      ipl->height = height;
      ipl->widthStep = min_width_step;
      ipl->nChannels = nchannels;
      return;
    }
    ReleaseImage(image);   // Not enough memory: start fresh.
  }

  // cvCreateImage will not return upon error in default error handling
  // mode CV_ErrModeLeaf.  CHECK in case it does return.
  /*VLOG(1) << Serial() << " WImageBuffer::Allocate "
          << width << "*" << height << "*" << nchannels << "*" << sizeofT
          << " = " << datasize;
  CHECK_GE(static_cast<int>(datasize), 0)
      << "OpenCV can't allocate image larger than 1<<31 in memory."
      << " (that's a bug in OpenCV).";*/
  *image = cvCreateImage(cvSize(width, height), depth, nchannels);
 /* CHECK(*image !=  NULL) << "InitImageHeader failed: "
    << "width: " << width << ", height: " << height
    << ", channels: " << nchannels;*/
}

void WImageDataUtil::LogAdopt(WImageData* image) {
  if (image) {
    const int sizeofT = (image->depth & ~IPL_DEPTH_SIGN) / 8;
    const size_t datasize = image->width*image->height*image->nChannels*sizeofT;
   /* VLOG(1) << Serial() << " WImageBuffer::Adopt "
            << image->width << "*" << image->height
            << "*" << image->nChannels << "*" << sizeofT
            << " = " << datasize;*/
  }
}

void WImageDataUtil::ReleaseImage(WImageData** image) {
  IplImage *ipl = *image;
  if (ipl != NULL) {
    const int sizeofT = (ipl->depth & ~IPL_DEPTH_SIGN) / 8;
    const size_t old_datasize = ipl->width*ipl->height*ipl->nChannels*sizeofT;
    /*std::cout<< " WImageBuffer::ReleaseImage "
            << ipl->width << "*" << ipl->height
            << "*" << ipl->nChannels << "*" << sizeofT
            << " = -" << old_datasize << std::endl;*/
    cvReleaseImage(image);
  }
}

bool WImageDataUtil::InitImageHeader(int width, int height,
                                     int num_channels, int depth,
                                     WImageData* header) {
  const int sizeofT = (depth & ~WIMAGE_DEPTH_SIGN) / 8;
  int stride;
  if (!SafeMultiply(width, num_channels * sizeofT, &stride)) {
    return false;
  }
  if (!SafeMultiply(stride, height, NULL)) {
    return false;
  }
  IplImage* result = cvInitImageHeader(header, cvSize(width, height),
                                       depth, num_channels);
  if (result == NULL) {
   /* LOG(ERROR) << "InitImageHeader failed: "
               << "width: << " << width << ", height: " << height
               << "channels: " << num_channels;*/
    return false;
  }
  return true;
}

#endif

