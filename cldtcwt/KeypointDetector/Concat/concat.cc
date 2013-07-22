// Copyright (C) 2013 Timothy Gale
#include "concat.h"
#include "kernel.h"

using namespace ConcatNS;

#include <string>
#include <sstream>
#include <iostream>
#include <iterator>
#include <algorithm>

#include <stdexcept>
Concat::Concat(cl::Context& context,
               const std::vector<cl::Device>& devices)
   : context_(context)
{
    // The OpenCL kernel:
    std::ostringstream kernelInput;

    // Define some constants
   
    // Get input from the source file
    const char* fileText = reinterpret_cast<const char*> (kernel_cl);
    size_t fileTextLength = kernel_cl_len;

    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(std::make_pair(fileText, fileTextLength));

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
        
    // ...and extract the useful part, i.e. the kernel
    kernel_ = cl::Kernel(program, "concat");
}




void Concat::operator() 
      (cl::CommandQueue& commandQueue,
       cl::Buffer& inputArray, cl::Buffer& outputArray,
       cl::Buffer& cumCounts, size_t cumCountsIndex,
       size_t numFloatsPerItem,
       const std::vector<cl::Event>& waitEvents,
       cl::Event* doneEvent)
{
    // inputArray contains a list of items, each one numFloatsPerItem long.
    // cumCounts[cumCountsIndex] is the location in outputArray where the list
    // should start going, and cumCounts[cumCountsIndex+1] where the next list
    // should start.
    //
    // The command will not start until all of waitEvents have completed, and
    // once done will flag doneEvent.

    // Set all the arguments
    kernel_.setArg(0, inputArray);
    kernel_.setArg(1, outputArray);
    kernel_.setArg(2, cumCounts);
    kernel_.setArg(3, cl_uint(cumCountsIndex));
    kernel_.setArg(4, cl_uint(numFloatsPerItem));

    // Execute
    commandQueue.enqueueNDRangeKernel(kernel_, cl::NullRange,
                                      {1024, 1},
                                      {256, 1},
                                      &waitEvents, doneEvent);
}


