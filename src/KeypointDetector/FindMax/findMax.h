#ifndef FINDMAX_H
#define FINDMAX_H

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"
#include <vector>



class FindMax {
// Class that finds the local maxima above a threshold in an image

public:

    FindMax() = default;
    FindMax(const FindMax&) = default;
    FindMax(cl::Context& context,
           const std::vector<cl::Device>& devices);

    // The filter operation
    void operator() (cl::CommandQueue& commandQueue,
       cl::Image& input,        float inputScale,
       cl::Image& inputFiner,   float finerScale,
       cl::Image& inputCoarser, float coarserScale,
       float threshold, float eigenRatioThreshold,
       cl::Buffer& output,
       cl::Buffer& numOutputs,
       unsigned int numOutputsOffset,
       const std::vector<cl::Event>& waitEvents = std::vector<cl::Event>(),
       cl::Event* doneEvent = nullptr);

    size_t getPosLength() const;
    // Returns the number of floats included in each output.  At the moment, that
    // is (x, y, scale, --), so 4.

private:
    cl::Context context_;
    cl::Kernel kernel_;

    static const int wgSizeX_ = 16;
    static const int wgSizeY_ = 16;

    // Number of floats long to make each output position.  Comes in format
    // x, y, scale.
    const size_t posLen_ = 4;
};



#endif

