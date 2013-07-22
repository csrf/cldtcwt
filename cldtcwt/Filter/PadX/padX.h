// Copyright (C) 2013 Timothy Gale
#ifndef PADX_H
#define PADX_H


#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"


#include "Filter/imageBuffer.h"


class PadX {
    // Straightforward convolution along the x axis, with an odd-
    // lengthed set of coefficients.  The images are padded.

public:

    PadX() = default;
    PadX(const PadX&) = default;
    PadX(cl::Context& context, 
         const std::vector<cl::Device>& devices);

    void operator() (cl::CommandQueue& cq, 
                     ImageBuffer<cl_float>& image,
                     const std::vector<cl::Event>& waitEvents
                        = std::vector<cl::Event>(),
                     cl::Event* doneEvent = nullptr);

private:

    cl::Context context_;
    cl::Kernel kernel_;
    cl::Buffer filter_;

    static const size_t padding_ = 16;

};



#endif

