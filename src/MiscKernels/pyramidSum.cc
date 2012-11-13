#include "pyramidSum.h"
#include "util/clUtil.h"
#include <string>
#include <iostream>
#include "MiscKernels/pyramidSumKernel.h"


PyramidSum::PyramidSum(cl::Context& context, 
                                 const std::vector<cl::Device>& devices)
{
    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(
        std::make_pair(reinterpret_cast<const char*>
                          (src_MiscKernels_pyramidSumKernel_h_src), 
                       src_MiscKernels_pyramidSumKernel_h_src_len)
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
    kernel_ = cl::Kernel(program, "pyramidSum");
}



void PyramidSum::operator() (cl::CommandQueue& cq,
                             cl::Image& input1, float gain1,
                             cl::Image& input2, float gain2,
                             cl::Image& output,
                             const std::vector<cl::Event>& waitEvents,
                             cl::Event* doneEvent)
{
    const int wgSize = 16;

    cl::NDRange workgroupSize = {wgSize, wgSize};

    cl::NDRange globalSize = {
        roundWGs(input1.getImageInfo<CL_IMAGE_WIDTH>(), wgSize), 
        roundWGs(input1.getImageInfo<CL_IMAGE_HEIGHT>(), wgSize)
    }; 


    // Set all the arguments
    kernel_.setArg(0, sizeof(input1), &input1);
    kernel_.setArg(1, (gain1));
    kernel_.setArg(2, sizeof(input2), &input2);
    kernel_.setArg(3, (gain2));
    kernel_.setArg(4, sizeof(output), &output);

    // Execute
    cq.enqueueNDRangeKernel(kernel_, cl::NullRange,
                            globalSize, workgroupSize,
                            &waitEvents, doneEvent);
}


