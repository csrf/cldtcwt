#ifndef CLUTILCV_H
#define CLUTILCV_H


#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"

#include <highgui.h>

cl::Image2D createImage2D(cl::Context& context, cv::Mat& mat);

#endif

