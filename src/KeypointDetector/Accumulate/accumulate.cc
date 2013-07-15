// Copyright (C) 2013 Timothy Gale
#include "accumulate.h"
#include <iostream>
#include "util/clUtil.h"

#include "kernel.h"
using namespace AccumulateNS;

Accumulate::Accumulate(cl::Context& context, 
                       const std::vector<cl::Device>& devices)
{
    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(std::make_pair(
        reinterpret_cast<const char*>(kernel_cl),
        kernel_cl_len));

    // Compile it...
    cl::Program program(context, source);
    try {
        program.build(devices);
    } catch(cl::Error err) {
	    std::cerr 
		    << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
		    << std::endl;
	    throw;
    } 
        
    // ...and extract the useful part, viz the kernel
    kernel_ = cl::Kernel(program, "accumulate");
}



void Accumulate::operator() (cl::CommandQueue& cq, cl::Buffer& input,
                                            cl::Buffer& cumSum,
                                            cl_uint maxSum,
                      const std::vector<cl::Event>& waitEvents,
                      cl::Event* doneEvent)
{
    // Both input buffers should contain cl_uint's.  cumSum should be one longer
    // than input.

    // Set all the arguments
    kernel_.setArg(0, sizeof(input), &input);
    kernel_.setArg(1, 
            cl_uint(input.getInfo<CL_MEM_SIZE>() / sizeof(cl_uint)));
    kernel_.setArg(2, sizeof(cumSum), &cumSum);
    kernel_.setArg(3, cl_uint(maxSum));
    
    cq.enqueueTask(kernel_, &waitEvents, doneEvent);
}


