#ifndef QUADTOCOMPLEX_H
#define QUADTOCOMPLEX_H


#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"


#include "../imageBuffer.h"


class QuadToComplex {
    // Straightforward convolution along the x axis, with an odd-
    // lengthed set of coefficients.  The images are padded.

public:

    QuadToComplex() = default;
    QuadToComplex(const QuadToComplex&) = default;
    QuadToComplex(cl::Context& context, 
            const std::vector<cl::Device>& devices);

    void operator() (cl::CommandQueue& cq, ImageBuffer& input,
                     cl::Image2D& output0,
                     cl::Image2D& output1,
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

