#include "absToRGBA.h"
#include "util/clUtil.h"
#include <string>
#include <iostream>

#include "kernel.h"
using namespace AbsToRGBANS;


AbsToRGBA::AbsToRGBA(cl::Context& context, 
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
    kernel_ = cl::Kernel(program, "absToRGBA");
}



void AbsToRGBA::operator() (cl::CommandQueue& cq, 
                            ImageBuffer<Complex<cl_float>>& input,
                            cl::Image& output,
                            float gain,
                            const std::vector<cl::Event>& waitEvents,
                            cl::Event* doneEvent)
{
    const int wgSize = 16;

    cl::NDRange workgroupSize = {wgSize, wgSize};

    cl::NDRange globalSize = {
        roundWGs(output.getImageInfo<CL_IMAGE_WIDTH>(), wgSize), 
        roundWGs(output.getImageInfo<CL_IMAGE_HEIGHT>(), wgSize)
    }; 


    // Set all the arguments
    kernel_.setArg(0, input.buffer());
    kernel_.setArg(1, cl_uint(input.start()));
    kernel_.setArg(2, cl_uint(input.padding()));
    kernel_.setArg(3, cl_uint(input.stride()));
    kernel_.setArg(4, sizeof(output), &output);
    kernel_.setArg(5, cl_float(gain));

    // Execute
    cq.enqueueNDRangeKernel(kernel_, cl::NullRange,
                            globalSize, workgroupSize,
                            &waitEvents, doneEvent);
}


