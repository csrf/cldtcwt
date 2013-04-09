#include "padY.h"
#include "util/clUtil.h"
#include <sstream>
#include <string>
#include <iostream>
#include <cassert>

#include "kernel.h"

using namespace PadYNS;

PadY::PadY(cl::Context& context, 
           const std::vector<cl::Device>& devices)
{
    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(
        std::make_pair(reinterpret_cast<const char*>(kernel_cl), 
                       kernel_cl_len)
    );

    std::ostringstream compilerOptions;
    compilerOptions << "-D PADDING=" << padding_;

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
    kernel_ = cl::Kernel(program, "padY");

}



void PadY::operator() (cl::CommandQueue& cq, 
                       ImageBuffer<cl_float>& image, 
                       const std::vector<cl::Event>& waitEvents,
                       cl::Event* doneEvent)
{
    // Padding etc.
    cl::NDRange workgroupSize = {padding_, padding_};

    cl::NDRange globalSize = {
        roundWGs(image.width(), workgroupSize[0]),
        2 * workgroupSize[1] 
    }; 

    // Must have the padding the kernel expects
    assert(image.padding() == padding_);

    // Set all the arguments
    kernel_.setArg(0, image.buffer());
    kernel_.setArg(1, cl_uint(image.start()));
    kernel_.setArg(2, cl_uint(image.height()));
    kernel_.setArg(3, cl_uint(image.stride()));

    // Execute
    cq.enqueueNDRangeKernel(kernel_, {0,0},
                            globalSize, workgroupSize,
                            &waitEvents, doneEvent);
}


