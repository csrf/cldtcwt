#include "copyImage.h"
#include "util/clUtil.h"
#include <sstream>
#include <string>
#include <iostream>
#include <cassert>

#include "kernel.h"

using namespace CopyImageNS;

CopyImage::CopyImage(cl::Context& context, 
                 const std::vector<cl::Device>& devices,
                 size_t padding,
                 size_t workgroupSize)
 : padding_(padding),
   workgroupSize_(workgroupSize)
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
    kernel_ = cl::Kernel(program, "copyImage");

}



void CopyImage::operator() (cl::CommandQueue& cq, 
                 ImageBuffer<cl_float>& input, 
                 ImageBuffer<cl_float>& output,
                 const std::vector<cl::Event>& waitEvents,
                 cl::Event* doneEvent)
{
    // Padding etc.
    cl::NDRange workgroupSize = {workgroupSize_, workgroupSize_};
    cl::NDRange offset = {padding_, padding_};

    cl::NDRange globalSize = {
        roundWGs(output.width(), workgroupSize[0]), 
        roundWGs(output.height(), workgroupSize[1])
    }; 

    // Must have the padding the kernel expects
    assert(input.padding() == padding_);

    // Input and output formats need to be exactly the same
    assert(input.width() == output.width());
    assert(input.height() == output.height());
    assert(input.stride() == output.stride());
    assert(input.padding() == output.padding());
    

    // Set all the arguments
    kernel_.setArg(0, input.buffer());
    kernel_.setArg(1, output.buffer());
    kernel_.setArg(2, int(input.width()));
    kernel_.setArg(3, int(input.stride()));

    // Execute
    cq.enqueueNDRangeKernel(kernel_, offset,
                            globalSize, workgroupSize,
                            &waitEvents, doneEvent);
}


