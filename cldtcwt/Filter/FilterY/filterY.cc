// Copyright (C) 2013 Timothy Gale
#include "filterY.h"
#include "util/clUtil.h"
#include <sstream>
#include <string>
#include <iostream>
#include <cassert>

#include "kernel.h"

using namespace FilterYNS;

FilterY::FilterY(cl::Context& context, 
                 const std::vector<cl::Device>& devices,
                 std::vector<float> filter)
{
    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(
        std::make_pair(reinterpret_cast<const char*>(kernel_cl), 
                       kernel_cl_len)
    );

    std::ostringstream compilerOptions;
    compilerOptions << "-D WG_W=" << workgroupSize_ << " "
                    << "-D WG_H=" << workgroupSize_ << " "
                    << "-D FILTER_LENGTH=" << filter.size() << " "
                    << "-D PADDING=" << padding_;

    // Compile it...
    cl::Program program(context, source);
    try {
        program.build(devices, compilerOptions.str().c_str());
    } catch(cl::Error err) {
	    std::cerr 
		    << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
		    << std::endl;
	    throw;
    } 
        
    // ...and extract the useful part, viz the kernel
    kernel_ = cl::Kernel(program, "filterY");


    // Upload the filter coefficients
    filter_ = cl::Buffer(context,
                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         filter.size() * sizeof(float),
                         &filter[0]);
    filterLength_ = filter.size();

    // Set that filter for use
    kernel_.setArg(5, filter_);

    // Make sure the filter is odd-length
    assert((filterLength_ & 1) == 1);

    // Make sure the filter is short enough that we can load
    // all the necessary surrounding data with the kernel
    assert((filterLength_-1) / 2 <= workgroupSize_ / 2);

    // Make sure we have enough padding to load the adjacent
    // values without going out of the image
    assert(padding_ >= workgroupSize_ / 2);
}



void FilterY::operator() (cl::CommandQueue& cq, 
                 ImageBuffer<cl_float>& input, 
                 ImageBuffer<cl_float>& output,
                 const std::vector<cl::Event>& waitEvents,
                 cl::Event* doneEvent)
{
    // Padding etc.
    cl::NDRange workgroupSize = {workgroupSize_, workgroupSize_};

    cl::NDRange globalSize = {
        roundWGs(output.width(), workgroupSize[0]), 
        roundWGs(output.height(), workgroupSize[1])
    }; 

    // Must have the padding the kernel expects
    assert(input.padding() == padding_);
    // This could actually be less: depends on the filter length

    // Input and output formats need to be exactly the same
    assert(input.width() == output.width());
    assert(input.height() == output.height());
    assert(input.stride() == output.stride());
    

    // Set all the arguments
    kernel_.setArg(0, input.buffer());
    kernel_.setArg(1, cl_uint(input.start()));
    kernel_.setArg(2, cl_uint(input.stride()));

    kernel_.setArg(3, output.buffer());
    kernel_.setArg(4, cl_uint(output.start()));

    // Execute
    cq.enqueueNDRangeKernel(kernel_, {0,0},
                            globalSize, workgroupSize,
                            &waitEvents, doneEvent);
}


