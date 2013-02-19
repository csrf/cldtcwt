#include "greyscaleToRGBA.h"
#include "util/clUtil.h"
#include <string>
#include <iostream>

#include "kernel.h"
using namespace GreyscaleToRGBANS;


GreyscaleToRGBA::GreyscaleToRGBA(cl::Context& context, 
                                 const std::vector<cl::Device>& devices)
{
    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(
        std::make_pair(reinterpret_cast<const char*> (kernel_cl), 
                       kernel_cl_len)
    );

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
    kernel_ = cl::Kernel(program, "greyscaleToRGBA");
}



void GreyscaleToRGBA::operator() (cl::CommandQueue& cq, cl::Image& input,
                                       cl::Image& output,
                                       float gain,
                 const std::vector<cl::Event>& waitEvents,
                 cl::Event* doneEvent)
{
    // Process, multiplying the final result by the gain
    const int wgSize = 16;

    cl::NDRange workgroupSize = {wgSize, wgSize};

    cl::NDRange globalSize = {
        roundWGs(output.getImageInfo<CL_IMAGE_WIDTH>(), wgSize), 
        roundWGs(output.getImageInfo<CL_IMAGE_HEIGHT>(), wgSize)
    }; 


    // Set all the arguments
    kernel_.setArg(0, sizeof(input), &input);
    kernel_.setArg(1, sizeof(output), &output);
    kernel_.setArg(2, float(gain));

    // Execute
    cq.enqueueNDRangeKernel(kernel_, cl::NullRange,
                            globalSize, workgroupSize,
                            &waitEvents, doneEvent);
}


