// Copyright (C) 2013 Timothy Gale
#ifndef PYRAMID_SUM_H
#define PYRAMID_SUM_H


#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"


class PyramidSum {
    // Kernel that takes two images, the second at half the resolution
    // of the first, and adds them up, using linear interpolation
    // on the second

public:

    PyramidSum() = default;
    PyramidSum(const PyramidSum&) = default;
    PyramidSum(cl::Context& context, 
               const std::vector<cl::Device>& devices);

    void operator() (cl::CommandQueue& cq, 
                     cl::Image& input1, float gain1,
                     cl::Image& input2, float gain2,
                     cl::Image& output,
                     const std::vector<cl::Event>& waitEvents
                        = std::vector<cl::Event>(),
                     cl::Event* doneEvent = nullptr);

private:

    cl::Context context_;
    cl::Kernel kernel_;

};



#endif

