#include "imageToImageBuffer.h"
#include "util/clUtil.h"
#include <sstream>
#include <string>
#include <iostream>
#include <cassert>

#include "kernel.h"

using namespace ImageToImageBufferNS;

ImageToImageBuffer::ImageToImageBuffer(cl::Context& context, 
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
    kernel_ = cl::Kernel(program, "imageToImageBuffer");
}




void ImageToImageBuffer::operator() (cl::CommandQueue& cq, 
                 cl::Image2D& input,
                 ImageBuffer<cl_float>& output,
                 const std::vector<cl::Event>& waitEvents,
                 cl::Event* doneEvent)
{
    // Padding etc.
    cl::NDRange workgroupSize = {workgroupSize_, workgroupSize_};
    cl::NDRange offset = {0, 0};

    cl::NDRange globalSize = {
        roundWGs(output.width(), workgroupSize[0]), 
        roundWGs(output.height(), workgroupSize[1])
    }; 

    // Input and output formats need to be compatible
    assert(output.width() == input.getImageInfo<CL_IMAGE_WIDTH>());
    assert(output.height() == input.getImageInfo<CL_IMAGE_HEIGHT>());

    // Set all the arguments
    kernel_.setArg(0, input);

    // Output buffer
    kernel_.setArg(1, output.buffer());
    kernel_.setArg(2, cl_uint(output.padding() * (1 + output.stride())));
    kernel_.setArg(3, cl_uint(output.stride()));

    // Execute
    cq.enqueueNDRangeKernel(kernel_, offset,
                            globalSize, workgroupSize,
                            &waitEvents, doneEvent);
}


