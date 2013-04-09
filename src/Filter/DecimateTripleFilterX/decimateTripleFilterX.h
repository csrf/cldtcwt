#ifndef DECIMATE_TRIPLE_FILTERX_H
#define DECIMATE_TRIPLE_FILTERX_H


#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"


#include "Filter/imageBuffer.h"


class DecimateTripleFilterX {
    // Decimated convolution along the x axis, with an even-
    // lengthed set of coefficients.  The images must be padded,
    // with alignment of twice the workgroup size.
    //
    // Three sets of coefficents are provided, and three outputs 
    // produced for every input.  Useful for reducing the number of
    // reads from the input image.

public:

    DecimateTripleFilterX() = default;
    DecimateTripleFilterX(const DecimateTripleFilterX&) = default;
    DecimateTripleFilterX(cl::Context& context, 
            const std::vector<cl::Device>& devices,
            std::vector<float> filter0, bool swapPairOrder0,
            std::vector<float> filter1, bool swapPairOrder1,
            std::vector<float> filter2, bool swapPairOrder2);
    // filter is the set of coefficients to convolve with the first of
    // the pair of trees forwards, and the second backwards.  The order
    // these trees are interleaved in the output is reversed if
    // swapPairOrder is true.  filter must be even length.  The output
    // order can be swaped by setting the corresponding flag true.
    // Three outputs are produced by the same kernel.

    void operator() (cl::CommandQueue& cq, 
                     ImageBuffer<cl_float>& input,
                     ImageBuffer<cl_float>& output,
                     const std::vector<cl::Event>& waitEvents
                        = std::vector<cl::Event>(),
                     cl::Event* doneEvent = nullptr);

private:

    cl::Context context_;
    cl::Kernel kernel_;
    cl::Buffer filter0_;
    cl::Buffer filter1_;
    cl::Buffer filter2_;

    size_t filterLength_;

    static const size_t padding_ = 16,
                        alignment_ = 32,
                        workgroupSize_ = 16;

};



#endif

