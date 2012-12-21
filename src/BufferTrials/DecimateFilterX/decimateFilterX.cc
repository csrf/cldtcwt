#include "decimateFilterX.h"
#include "util/clUtil.h"
#include <sstream>
#include <string>
#include <iostream>
#include <cassert>

#include "BufferTrials/DecimateFilterX/kernel.h"


DecimateFilterX::DecimateFilterX(cl::Context& context, 
                 const std::vector<cl::Device>& devices,
                 std::vector<float> filter,
                 bool swapOutputPair)
{
    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(
        std::make_pair(reinterpret_cast<const char*>
              (src_BufferTrials_DecimateFilterX_kernel_h_src), 
               src_BufferTrials_DecimateFilterX_kernel_h_src_len)
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
    kernel_ = cl::Kernel(program, "decimateFilterX");

    // We want the forward and backward coefficients interleaved:
    // going back first, then forward
    std::vector<float> interleavedFilter(2*filter.size());

    filterLength_ = filter.size();
    for (int n = 0; n < filter.size(); ++n) {
        interleavedFilter[2*n] = filter[filterLength_ - n - 1];
        interleavedFilter[2*n+1] = filter[n];
    }

    // Upload the filter coefficients
    filter_ = cl::Buffer(context,
                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         interleavedFilter.size() * sizeof(float),
                         &interleavedFilter[0]);

    // Set that filter for use
    kernel_.setArg(2, filter_);

    // Make sure the filter is even-length
    assert((filterLength_ & 1) == 0);

    // Make sure the filter is short enough that we can load
    // all the necessary surrounding data with the kernel (plus
    // an extra if we need an extension)
    assert(filterLength_-1 <= workgroupSize_);

    // Make sure we have enough padding to load the adjacent
    // values without going out of the image to the left/top
    assert(padding_ >= workgroupSize_);
}



void DecimateFilterX::operator() (cl::CommandQueue& cq, 
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

    // Input and output formats need to be exactly the same
    assert(input.width()+2 >= 2*output.width());
    assert(input.height() == output.height());
    assert(input.padding() == output.padding());
    

    // Set all the arguments
    kernel_.setArg(0, input.buffer());
    kernel_.setArg(1, output.buffer());
    kernel_.setArg(3, int(input.width()));
    kernel_.setArg(4, int(input.stride()));
    kernel_.setArg(5, int(output.stride()));

    // Execute
    cq.enqueueNDRangeKernel(kernel_, offset,
                            globalSize, workgroupSize,
                            &waitEvents, doneEvent);
}


