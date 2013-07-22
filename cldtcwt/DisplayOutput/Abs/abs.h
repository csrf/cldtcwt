// Copyright (C) 2013 Timothy Gale
#ifndef ABS_H
#define ABS_H


#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"


class Abs {
    // Kernel that takes a two-component (i.e. complex) image, and puts out a
    // single-component (magnitude) image

public:

    Abs() = default;
    Abs(const Abs&) = default;
    Abs(cl::Context& context, const std::vector<cl::Device>& devices);

    void operator() (cl::CommandQueue& cq, cl::Image& input,
                                           cl::Image& output,
                     const std::vector<cl::Event>& waitEvents
                        = std::vector<cl::Event>(),
                     cl::Event* doneEvent = nullptr);

private:

    cl::Context context_;
    cl::Kernel kernel_;

};



#endif

