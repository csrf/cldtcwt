#ifndef DECIMATE_FILTERX_H
#define DECIMATE_FILTERX_H


#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"


#include "Filter/imageBuffer.h"


class DecimateFilterX_alt {
    // Decimated convolution along the x axis, with an even-
    // lengthed set of coefficients.  The images must be padded,
    // with alignment of twice the workgroup size.

public:

    DecimateFilterX_alt() = default;
    DecimateFilterX_alt(const DecimateFilterX_alt&) = default;
    DecimateFilterX_alt(cl::Context& context, 
            const std::vector<cl::Device>& devices,
            std::vector<float> filter,
            bool swapPairOrder);
    // filter is the set of coefficients to convolve with the first of
    // the pair of trees forwards, and the second backwards.  The order
    // these trees are interleaved in the output is reversed if
    // swapPairOrder is true.  filter must be even length.

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
                        alignment_ = 32,
                        workgroupSize_ = 16;

};



#endif

