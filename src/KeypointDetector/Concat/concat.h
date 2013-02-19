#ifndef CONCAT_H
#define CONCAT_H

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"
#include <vector>



class Concat {
// Class that finds the local maxima above a threshold in an image

public:

    Concat() = default;
    Concat(const Concat&) = default;
    Concat(cl::Context& context,
           const std::vector<cl::Device>& devices);

    // The kernel operation
    void operator() (cl::CommandQueue& commandQueue,
       cl::Buffer& inputArray, cl::Buffer& outputArray,
       cl::Buffer& cumCounts, size_t cumCountsIndex,
       size_t numFloatsPerItem,
       const std::vector<cl::Event>& waitEvents = std::vector<cl::Event>(),
       cl::Event* doneEvent = nullptr);

private:
    cl::Context context_;
    cl::Kernel kernel_;
};



#endif

