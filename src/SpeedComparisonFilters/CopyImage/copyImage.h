// Copyright (C) 2013 Timothy Gale
#ifndef FILTERX_H
#define FILTERX_H


#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"


#include "Filter/imageBuffer.h"




class CopyImage {
    // Straightforward convolution along the x axis, with an odd-
    // lengthed set of coefficients.  The images are padded.

public:

    CopyImage() = default;
    CopyImage(const CopyImage&) = default;
    CopyImage(cl::Context& context, 
            const std::vector<cl::Device>& devices,
            size_t padding = 16,
            size_t workgroupSize = 16);

    void operator() (cl::CommandQueue& cq, ImageBuffer<cl_float>& input,
                                           ImageBuffer<cl_float>& output,
                     const std::vector<cl::Event>& waitEvents
                        = std::vector<cl::Event>(),
                     cl::Event* doneEvent = nullptr);

private:

    cl::Context context_;
    cl::Kernel kernel_;

    size_t padding_;
    size_t workgroupSize_;

};



#endif

