#include "quadToComplex.h"
#include "util/clUtil.h"
#include <sstream>
#include <string>
#include <iostream>
#include <cassert>

#include "kernel.h"

using namespace QuadToComplexNS;

QuadToComplex::QuadToComplex(cl::Context& context, 
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
                    << "-D WG_H=" << workgroupSize_ << " "
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
    kernel_ = cl::Kernel(program, "quadToComplex");
}




void QuadToComplex::operator() (cl::CommandQueue& cq, 
                 ImageBuffer<cl_float>& input, 
                 cl::Image2D& output0, cl::Image2D& output1,
                 const std::vector<cl::Event>& waitEvents,
                 cl::Event* doneEvent)
{
    // Padding etc.
    cl::NDRange workgroupSize = {workgroupSize_, workgroupSize_};
    cl::NDRange offset = {0, 0};

    cl::NDRange globalSize = {
        roundWGs(input.width(), workgroupSize[0]), 
        roundWGs(input.height(), workgroupSize[1])
    }; 

    // Must have the padding the kernel expects
    assert(input.padding() == padding_);

    // Input and output formats need to be compatible
    assert(input.width() == 2*output0.getImageInfo<CL_IMAGE_WIDTH>());
    assert(input.height() == 2*output0.getImageInfo<CL_IMAGE_HEIGHT>());
    assert(input.width() == 2*output1.getImageInfo<CL_IMAGE_WIDTH>());
    assert(input.height() == 2*output1.getImageInfo<CL_IMAGE_HEIGHT>());

    // Set all the arguments
    kernel_.setArg(0, input.buffer());
    kernel_.setArg(1, int(input.stride()));
    kernel_.setArg(2, output0);
    kernel_.setArg(3, output1);

    // Execute
    cq.enqueueNDRangeKernel(kernel_, offset,
                            globalSize, workgroupSize,
                            &waitEvents, doneEvent);
}


