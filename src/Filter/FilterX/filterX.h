#ifndef FILTERX_H
#define FILTERX_H


#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"


#include "Filter/imageBuffer.h"




class FilterX {
    // Straightforward convolution along the x axis, with an odd-
    // lengthed set of coefficients.  The images are padded.

public:

    FilterX() = default;
    FilterX(const FilterX&) = default;
    FilterX(cl::Context& context, 
            const std::vector<cl::Device>& devices,
            std::vector<float> filter);

    void operator() (cl::CommandQueue& cq, ImageBuffer<cl_float>& input,
                                           ImageBuffer<cl_float>& output,
                     const std::vector<cl::Event>& waitEvents
                        = std::vector<cl::Event>(),
                     cl::Event* doneEvent = nullptr);

private:

    cl::Context context_;
    cl::Kernel kernel_;
    cl::Buffer filter_;

    size_t filterLength_;

    static const size_t padding_ = 16,
                        workgroupSize_ = 16;

};



#endif

