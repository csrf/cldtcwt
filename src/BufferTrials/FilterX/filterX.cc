#include "filterX.h"
#include "util/clUtil.h"
#include <sstream>
#include <string>
#include <iostream>
#include <cassert>

#include "BufferTrials/FilterX/filterXKernel.h"


FilterX::FilterX(cl::Context& context, 
                 const std::vector<cl::Device>& devices,
                 std::vector<float> filter)
{
    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(
        std::make_pair(reinterpret_cast<const char*>
              (src_BufferTrials_FilterX_filterXKernel_h_src), 
               src_BufferTrials_FilterX_filterXKernel_h_src_len)
    );

    std::ostringstream compilerOptions;
    compilerOptions << "-D WG_W=" << workgroupSize_ << " "
                    << "-D WG_H=" << workgroupSize_ << " "
                    << "-D FILTER_LENGTH=" << filter.size() << " "
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
    kernel_ = cl::Kernel(program, "filterX");


    // Upload the filter coefficients
    filter_ = cl::Buffer(context,
                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         filter.size() * sizeof(float),
                         &filter[0]);
    filterLength_ = filter.size();

    // Set that filter for use
    kernel_.setArg(2, filter_);

    // Make sure the filter is odd-length
    assert((filterLength_ & 1) == 1);

    // Make sure the filter is short enough that we can load
    // all the necessary surrounding data with the kernel
    assert((filterLength_-1) / 2 <= workgroupSize_ / 2);

    // Make sure we have enough padding to load the adjacent
    // values without going out of the image
    assert(padding_ >= workgroupSize_ / 2);
}



void FilterX::operator() (cl::CommandQueue& cq, 
                 ImageBuffer& input, ImageBuffer& output,
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

    // Must be big enough that any edge mirroring can be
    // handled by the filter
    assert(input.width() >= ((filterLength_ - 1) / 2));

    // Input and output formats need to be exactly the same
    assert(input.width() == output.width());
    assert(input.height() == output.height());
    assert(input.stride() == output.stride());
    assert(input.padding() == output.padding());
    

    // Set all the arguments
    kernel_.setArg(0, input.buffer());
    kernel_.setArg(1, output.buffer());
    kernel_.setArg(3, int(input.width()));
    kernel_.setArg(4, int(input.stride()));

    // Execute
    cq.enqueueNDRangeKernel(kernel_, offset,
                            globalSize, workgroupSize,
                            &waitEvents, doneEvent);
}


