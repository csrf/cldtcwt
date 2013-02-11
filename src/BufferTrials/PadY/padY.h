#ifndef PADY_H
#define PADY_H


#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"


#include "BufferTrials/imageBuffer.h"


class PadY {
    // Straightforward convolution along the x axis, with an odd-
    // lengthed set of coefficients.  The images are padded.

public:

    PadY() = default;
    PadY(const PadY&) = default;
    PadY(cl::Context& context, 
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

