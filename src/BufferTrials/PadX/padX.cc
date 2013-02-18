#include "padX.h"
#include "util/clUtil.h"
#include <sstream>
#include <string>
#include <iostream>
#include <cassert>

#include "kernel.h"

using namespace PadXNS;

PadX::PadX(cl::Context& context, 
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
    kernel_ = cl::Kernel(program, "padX");

}



void PadX::operator() (cl::CommandQueue& cq, 
                       ImageBuffer<cl_float>& image, 
                       const std::vector<cl::Event>& waitEvents,
                       cl::Event* doneEvent)
{
    // Padding etc.
    cl::NDRange workgroupSize = {padding_, padding_};
    cl::NDRange offset = {padding_, padding_};

    cl::NDRange globalSize = {
        2 * workgroupSize[0], 
        roundWGs(image.height(), workgroupSize[1])
    }; 

    // Must have the padding the kernel expects
    assert(image.padding() == padding_);

    // Set all the arguments
    kernel_.setArg(0, image.buffer());
    kernel_.setArg(1, int(image.width()));
    kernel_.setArg(2, int(image.stride()));

    // Execute
    cq.enqueueNDRangeKernel(kernel_, offset,
                            globalSize, workgroupSize,
                            &waitEvents, doneEvent);
}


