#ifndef ABSTORGBA_H
#define ABSTORGBA_H

#include "Filter/imageBuffer.h"

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"


class AbsToRGBA {
    // Kernel that takes a two-component (complex) image, and puts out an
    // RGBA (A=1) image that consists of its absolute value.

public:

    AbsToRGBA() = default;
    AbsToRGBA(const AbsToRGBA&) = default;
    AbsToRGBA(cl::Context& context, 
              const std::vector<cl::Device>& devices);

    void operator() (cl::CommandQueue& cq, 
                     ImageBuffer<Complex<cl_float>>& input,
                     cl::Image& output,
                     float gain = 1.0f,
                     const std::vector<cl::Event>& waitEvents
                        = std::vector<cl::Event>(),
                     cl::Event* doneEvent = nullptr);

private:

    cl::Context context_;
    cl::Kernel kernel_;

};



#endif

