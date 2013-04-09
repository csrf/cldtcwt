#include "decimateFilterX.h"
#include "util/clUtil.h"
#include <sstream>
#include <string>
#include <iostream>
#include <cassert>

#include "kernel.h"

using namespace DecimateFilterXNS;

DecimateFilterX::DecimateFilterX(cl::Context& context, 
                 const std::vector<cl::Device>& devices,
                 std::vector<float> filter,
                 bool swapOutputPair)
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
                    << "-D FILTER_LENGTH=" << filter.size() << " ";

    if (swapOutputPair)
        compilerOptions << "-D SWAP_TREE_1 ";

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

    // We want the filter to be reversed 
    std::vector<float> reversedFilter(filter.size());

    filterLength_ = filter.size();
    for (int n = 0; n < filter.size(); ++n) 
        reversedFilter[n] = filter[filterLength_ - n - 1];

    // Upload the filter coefficients
    filter_ = cl::Buffer(context,
                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         reversedFilter.size() * sizeof(float),
                         &reversedFilter[0]);

    // Set that filter for use
    kernel_.setArg(6, filter_);

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
                 ImageBuffer<cl_float>& input, 
                 ImageBuffer<cl_float>& output,
                 const std::vector<cl::Event>& waitEvents,
                 cl::Event* doneEvent)
{
    // Padding etc.
    cl::NDRange workgroupSize = {workgroupSize_, workgroupSize_};

    cl::NDRange globalSize = {
        roundWGs(output.width(), workgroupSize[0]), 
        roundWGs(output.height(), workgroupSize[1])
    }; 

    // Must have enough padding
    assert(input.padding() == padding_);

    // Pad symmetrically if needed
    bool symmetricPadding = output.width() * 2 > input.width();

    // Input and output formats need to be exactly the same
    assert((input.width() + symmetricPadding * 2) == 2*output.width());
    assert(input.height() == output.height());
    
    // Set all the arguments

    // Input buffer
    kernel_.setArg(0, input.buffer());
    kernel_.setArg(1, cl_uint(input.start() - symmetricPadding));
    kernel_.setArg(2, cl_uint(input.stride()));

    // Output buffer
    kernel_.setArg(3, output.buffer());
    kernel_.setArg(4, cl_uint(output.start()));
    kernel_.setArg(5, cl_uint(output.stride()));

    // Execute
    cq.enqueueNDRangeKernel(kernel_, {0, 0},
                            globalSize, workgroupSize,
                            &waitEvents, doneEvent);
}


