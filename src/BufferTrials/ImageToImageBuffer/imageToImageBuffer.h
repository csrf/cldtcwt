#ifndef IMAGE_TO_IMAGE_BUFFER_H
#define IMAGE_TO_IMAGE_BUFFER_H


#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"


#include "../imageBuffer.h"


class ImageToImageBuffer {
    // Straightforward convolution along the x axis, with an odd-
    // lengthed set of coefficients.  The images are padded.

public:

    ImageToImageBuffer() = default;
    ImageToImageBuffer(const ImageToImageBuffer&) = default;
    ImageToImageBuffer(cl::Context& context, 
            const std::vector<cl::Device>& devices);

    void operator() (cl::CommandQueue& cq, 
                     cl::Image2D& input,
                     ImageBuffer& output,
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

