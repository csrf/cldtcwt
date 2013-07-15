// Copyright (C) 2013 Timothy Gale
#ifndef SCALE_IMAGE_TO_IMAGE_BUFFER_H
#define SCALE_IMAGE_TO_IMAGE_BUFFER_H


#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"


#include "Filter/imageBuffer.h"


class ScaleImageToImageBuffer {
    // Straightforward convolution along the x axis, with an odd-
    // lengthed set of coefficients.  The images are padded.

public:

    ScaleImageToImageBuffer() = default;
    ScaleImageToImageBuffer(const ScaleImageToImageBuffer&) = default;
    ScaleImageToImageBuffer(cl::Context& context, 
            const std::vector<cl::Device>& devices);

    void operator() (cl::CommandQueue& cq, 
                     cl::Image2D& input,
                     ImageBuffer<cl_float>& output,
                     float scaleFactor,
                     const std::vector<cl::Event>& waitEvents
                        = std::vector<cl::Event>(),
                     cl::Event* doneEvent = nullptr);

private:

    cl::Context context_;
    cl::Kernel kernel_;

    static const size_t padding_ = 16,
                        workgroupSize_ = 16;

};



#endif

