#include "filter.h"
#include "util/clUtil.h"
#include <string>
#include <iostream>
#include "BufferTrials/filterXKernel.h"


FilterX::FilterX(cl::Context& context, 
                                 const std::vector<cl::Device>& devices)
{
    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(
        std::make_pair(reinterpret_cast<const char*>
                          (src_BufferTrials_filterXKernel_h_src), 
                       src_BufferTrials_filterXKernel_h_src_len)
    );

    // Compile it...
    cl::Program program(context, source);
    try {
        program.build(devices, "-D WG_W=16 -D WG_H=16 "
                               "-D FILTER_LENGTH=13");
    } catch(cl::Error err) {
	    std::cerr 
		    << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
		    << std::endl;
	    throw;
    } 
        
    // ...and extract the useful part, viz the kernel
    kernel_ = cl::Kernel(program, "filterX");
}



void FilterX::operator() (cl::CommandQueue& cq, 
                 ImageBuffer& input, ImageBuffer& output,
                 const std::vector<cl::Event>& waitEvents,
                 cl::Event* doneEvent)
{
    // Process, multiplying the final result by the gain
    const int wgSize = 16;

    cl::NDRange workgroupSize = {16, 16};

    cl::NDRange globalSize = {
        roundWGs(output.width, workgroupSize[0]), 
        roundWGs(output.height, workgroupSize[1])
    }; 


    // Set all the arguments
    kernel_.setArg(0, sizeof(input.buffer), &input.buffer);
    kernel_.setArg(1, sizeof(output.buffer), &output.buffer);
    kernel_.setArg(2, int(input.width));
    kernel_.setArg(3, int(input.stride));
    kernel_.setArg(4, int(input.height));

    // Execute
    cq.enqueueNDRangeKernel(kernel_, cl::NullRange,
                            globalSize, workgroupSize,
                            &waitEvents, doneEvent);
}


