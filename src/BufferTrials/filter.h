#ifndef FILTER_H
#define FILTER_H


#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"


struct ImageBuffer {
    cl::Buffer buffer;
    size_t width;
    size_t rowPadding;
    size_t stride;
    size_t height;
};




class FilterX {
    // Straightforward convolution along the x axis

public:

    FilterX() = default;
    FilterX(const FilterX&) = default;
    FilterX(cl::Context& context, 
            const std::vector<cl::Device>& devices,
            std::vector<float> filter);

    void operator() (cl::CommandQueue& cq, ImageBuffer& input,
                                           ImageBuffer& output,
                     const std::vector<cl::Event>& waitEvents
                        = std::vector<cl::Event>(),
                     cl::Event* doneEvent = nullptr);

private:

    cl::Context context_;
    cl::Kernel kernel_;
    cl::Buffer filter_;

};



#endif

