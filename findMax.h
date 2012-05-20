#ifndef FINDMAX_H
#define FINDMAX_H

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "cl.hpp"
#include <vector>




class FindMax {
// Class that finds the local maxima above a threshold in an image

public:

    FindMax(cl::Context& context,
           const std::vector<cl::Device>& devices);

    // The filter operation
    void operator() (cl::CommandQueue& commandQueue,
           const cl::Image2D& input,
           cl::Buffer& output,
           cl::Buffer& numOutputs,
           cl::Buffer& lock,
           const std::vector<cl::Event>& waitEvents = std::vector<cl::Event>(),
           cl::Event* doneEvent = nullptr);

private:
    cl::Context context_;
    cl::Kernel kernel_;

    const int wgSizeX_;
    const int wgSizeY_;
};



#endif

