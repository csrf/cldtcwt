#include "filter.h"
#include "util/clUtil.h"
#include <sstream>
#include <string>
#include <iostream>
#include "BufferTrials/filterXKernel.h"


FilterX::FilterX(cl::Context& context, 
                 const std::vector<cl::Device>& devices,
                 std::vector<float> filter)
{
    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(
        std::make_pair(reinterpret_cast<const char*>
                          (src_BufferTrials_filterXKernel_h_src), 
                       src_BufferTrials_filterXKernel_h_src_len)
    );

    std::ostringstream compilerOptions;
    compilerOptions << "-D WG_W=" << 16 << " "
                    << "-D WG_H=" << 16 << " "
                    << "-D FILTER_LENGTH=" << filter.size() << " "
                    << "-D PADDING=" << 8;

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
    kernel_ = cl::Kernel(program, "filterX");

    // Upload the filter coefficients
    filter_ = cl::Buffer(context,
                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         filter.size() * sizeof(float),
                         &filter[0]);

    // Set that filter for use
    kernel_.setArg(2, filter_);
}



void FilterX::operator() (cl::CommandQueue& cq, 
                 ImageBuffer& input, ImageBuffer& output,
                 const std::vector<cl::Event>& waitEvents,
                 cl::Event* doneEvent)
{
    // Process, multiplying the final result by the gain
    const int wgSize = 16;

    cl::NDRange workgroupSize = {wgSize, wgSize};

    // padding
    cl::NDRange offset = {8, 8};

    cl::NDRange globalSize = {
        roundWGs(output.width, workgroupSize[0]), 
        roundWGs(output.height, workgroupSize[1])
    }; 


    // Set all the arguments
    kernel_.setArg(0, sizeof(input.buffer), &input.buffer);
    kernel_.setArg(1, sizeof(output.buffer), &output.buffer);
    kernel_.setArg(3, int(input.width));
    kernel_.setArg(4, int(input.stride));
    kernel_.setArg(5, int(input.height));

    // Execute
    cq.enqueueNDRangeKernel(kernel_, offset,
                            globalSize, workgroupSize,
                            &waitEvents, doneEvent);
}


