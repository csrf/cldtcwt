#ifndef CLUTIL_H
#define CLUTIL_H


#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "cl.hpp"

#include <vector>



cl::Buffer createBuffer(cl::Context&, cl::CommandQueue&,
                        const std::vector<float>& data);

cl::Image2D createImage2D(cl::Context&, int width, int height);

void writeImage2D(cl::CommandQueue& commandQueue,
                  cl::Image2D& image, float* memory);

void readImage2D(cl::CommandQueue& commandQueue,
                 float* outMemory, cl::Image2D& image);


#endif
