#ifndef GREYSCALETORGBA_H
#define GREYSCALETORGBA_H


#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"


class GreyscaleToRGBA {
    // Kernel that takes a single-component image, and puts out an
    // RGBA (A=1) image.  Multiplies by the gain specified.

public:

    GreyscaleToRGBA() = default;
    GreyscaleToRGBA(const GreyscaleToRGBA&) = default;
    GreyscaleToRGBA(cl::Context& context, 
                    const std::vector<cl::Device>& devices);

    void operator() (cl::CommandQueue& cq, cl::Image& input,
                                           cl::Image& output,
                                           float gain,
                     const std::vector<cl::Event>& waitEvents
                        = std::vector<cl::Event>(),
                     cl::Event* doneEvent = nullptr);

private:

    cl::Context context_;
    cl::Kernel kernel_;

};



#endif

