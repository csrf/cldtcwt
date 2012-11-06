#ifndef ACCUMULATE_H
#define ACCUMULATE_H


#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"


class Accumulate {
    // Kernel that sums the values in a vector, giving zero as its first output.
    // The output should be one longer than the input.  maxSum provides a maximum
    // value for the output (e.g. in case the program has a maximum number of memory
    // slots it wants to copy into, and the cumulative total risks addressing more).

public:

    Accumulate() = default;
    Accumulate(const Accumulate&) = default;
    Accumulate(cl::Context& context, const std::vector<cl::Device>& devices);

    void operator() (cl::CommandQueue& cq, cl::Buffer& input,
                                           cl::Buffer& cumSum,
                                           cl_uint maxSum,
                     const std::vector<cl::Event>& waitEvents
                        = std::vector<cl::Event>(),
                     cl::Event* doneEvent = nullptr);

private:

    cl::Context context_;
    cl::Kernel kernel_;

};



#endif

