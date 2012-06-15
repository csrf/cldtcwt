#ifndef CLUTIL_H
#define CLUTIL_H


#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "cl.hpp"

#include <vector>

#include <highgui.h>


cl::Buffer createBuffer(cl::Context&, cl::CommandQueue&,
                        const std::vector<float>& data);

cl::Image2D createImage2D(cl::Context&, int width, int height);

cl::Image2D createImage2D(cl::Context& context, cv::Mat& mat);

void writeImage2D(cl::CommandQueue& commandQueue,
                  cl::Image2D& image, float* memory);

void readImage2D(cl::CommandQueue& commandQueue,
                 float* outMemory, cl::Image2D& image);

void saveRealImage(std::string filename,
                   cl::CommandQueue& cq, cl::Image2D& image);

void saveComplexImage(std::string filename,
                      cl::CommandQueue& cq, cl::Image2D& image);

void displayRealImage(cl::CommandQueue& cq, cl::Image2D& image);

#endif
