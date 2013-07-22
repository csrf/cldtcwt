// Copyright (C) 2013 Timothy Gale
#include "tripleQ2cDecimateFilterY.h"
#include "util/clUtil.h"
#include <sstream>
#include <string>
#include <iostream>
#include <cassert>
#include <iterator>
#include <algorithm>

#include "kernel.h"

using namespace TripleQuadToComplexDecimateFilterYNS;


TripleQuadToComplexDecimateFilterY::TripleQuadToComplexDecimateFilterY(cl::Context& context, 
                 const std::vector<cl::Device>& devices,
                 std::vector<float> filter0, bool swapOutputPair0,
                 std::vector<float> filter1, bool swapOutputPair1,
                 std::vector<float> filter2, bool swapOutputPair2)
    : filterLength_(filter0.size())
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
                    << "-D FILTER_LENGTH=" << filter0.size() << " ";

    // Swap tree pairs, if requested
    if (swapOutputPair0)
        compilerOptions << "-D SWAP_TREE_0 ";

    if (swapOutputPair1)
        compilerOptions << "-D SWAP_TREE_1 ";

    if (swapOutputPair2)
        compilerOptions << "-D SWAP_TREE_2 ";

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
    kernel_ = cl::Kernel(program, "decimateFilterY");

    // We want the filters to be reversed, and in the same vector (to 
    // reduce the number of arguments we need to pass in -- also, more
    // convenient to add an index-dependent offset than to switch between
    // three inputs)
    std::vector<cl_float> reversedFilters;

    std::copy(filter0.rbegin(), filter0.rend(), 
              std::back_inserter(reversedFilters));
    std::copy(filter1.rbegin(), filter1.rend(), 
              std::back_inserter(reversedFilters));
    std::copy(filter2.rbegin(), filter2.rend(), 
              std::back_inserter(reversedFilters));



    // Upload the filter coefficients
    filter_ = cl::Buffer {
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        reversedFilters.size() * sizeof(cl_float),
        &reversedFilters[0]
    };

    // Set that filter for use
    kernel_.setArg(10, filter_);

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



void TripleQuadToComplexDecimateFilterY::operator() (cl::CommandQueue& cq, 
                 ImageBuffer<cl_float>& input, 
                 ImageBuffer<Complex<cl_float>>& output,
                 const std::vector<cl::Event>& waitEvents,
                 cl::Event* doneEvent)
{
    // Padding etc.
    cl::NDRange workgroupSize = {workgroupSize_, workgroupSize_, 1};

    // Must have the padding the kernel expects
    assert(input.padding() == padding_);

    // Pad symmetrically if needed
    bool symmetricPadding = (input.height() % 4) == 2;

    // Input and output formats need to be compatible
    const size_t quadHeight = (input.height() + symmetricPadding * 2) / 2;
    assert(input.width() == 2 * output.width());
    assert(quadHeight == 2*output.height());

    cl::NDRange globalSize = {
        roundWGs(input.width(), workgroupSize[0]), 
        roundWGs(quadHeight, workgroupSize[1]),
        3
    }; 

    // Set all the arguments (other than the filter, which has already
    // been set)
    
    // Input buffer
    kernel_.setArg(0, input.buffer());
    kernel_.setArg(1, cl_uint(input.start() 
                                - symmetricPadding * input.stride()));
    kernel_.setArg(2, cl_uint(input.pitch()));
    kernel_.setArg(3, cl_uint(input.stride()));

    // Output buffers
    kernel_.setArg(4, output.buffer());
    kernel_.setArg(5, cl_uint(output.start()));
    kernel_.setArg(6, cl_uint(output.pitch()));
    kernel_.setArg(7, cl_uint(output.stride()));
    kernel_.setArg(8, cl_uint(output.width()));
    kernel_.setArg(9, cl_uint(output.height()));

    // Execute
    cq.enqueueNDRangeKernel(kernel_, {0, 0, 0},
                            globalSize, workgroupSize,
                            &waitEvents, doneEvent);
}


