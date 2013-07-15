// Copyright (C) 2013 Timothy Gale
#include "scaleImageToImageBuffer.h"
#include "util/clUtil.h"
#include <sstream>
#include <string>
#include <iostream>
#include <cassert>

#include "kernel.h"

using namespace ScaleImageToImageBufferNS;

ScaleImageToImageBuffer::ScaleImageToImageBuffer(cl::Context& context, 
                 const std::vector<cl::Device>& devices)
{
    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(
        std::make_pair(reinterpret_cast<const char*>(kernel_cl), 
                       kernel_cl_len)
    );

    std::ostringstream compilerOptions;
    compilerOptions << "-D WG_W=" << workgroupSize_ << " "
                    << "-D WG_H=" << workgroupSize_;

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
    kernel_ = cl::Kernel(program, "scaleImageToImageBuffer");
}




void ScaleImageToImageBuffer::operator() (cl::CommandQueue& cq, 
                 cl::Image2D& input,
                 ImageBuffer<cl_float>& output,
                 float scaleFactor,
                 const std::vector<cl::Event>& waitEvents,
                 cl::Event* doneEvent)
{
    // Padding etc.
    cl::NDRange workgroupSize = {workgroupSize_, workgroupSize_};

    cl::NDRange globalSize = {
        roundWGs(output.width(), workgroupSize[0]), 
        roundWGs(output.height(), workgroupSize[1])
    }; 

    // Output mustn't overwrite anything that isn't padding
    assert(output.padding() >= workgroupSize_);

    // Set all the arguments
    kernel_.setArg(0, input);

    // Output buffer
    kernel_.setArg(1, output.buffer());
    kernel_.setArg(2, cl_uint(output.start()));
    kernel_.setArg(3, cl_uint(output.stride()));

    // Centre of the output buffer
    kernel_.setArg(4, cl_float((output.width() - 1) / 2.f));
    kernel_.setArg(5, cl_float((output.height() - 1) / 2.f));

    // Inverse of the scale factor
    kernel_.setArg(6, cl_float(1.f / scaleFactor));

    // Execute
    cq.enqueueNDRangeKernel(kernel_, {0, 0},
                            globalSize, workgroupSize,
                            &waitEvents, doneEvent);
}


