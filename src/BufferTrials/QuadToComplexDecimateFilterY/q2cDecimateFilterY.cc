#include "q2cDecimateFilterY.h"
#include "util/clUtil.h"
#include <sstream>
#include <string>
#include <iostream>
#include <cassert>

#include "BufferTrials/QuadToComplexDecimateFilterY/kernel.h"


QuadToComplexDecimateFilterY::QuadToComplexDecimateFilterY(cl::Context& context, 
                 const std::vector<cl::Device>& devices,
                 std::vector<float> filter,
                 bool swapOutputPair)
{
    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(
        std::make_pair(reinterpret_cast<const char*>
              (src_BufferTrials_QuadToComplexDecimateFilterY_kernel_h_src), 
               src_BufferTrials_QuadToComplexDecimateFilterY_kernel_h_src_len)
    );

    std::ostringstream compilerOptions;
    compilerOptions << "-D WG_W=" << workgroupSize_ << " "
                    << "-D WG_H=" << workgroupSize_ << " "
                    << "-D FILTER_LENGTH=" << filter.size() << " "
                    << "-D PADDING=" << padding_ << " ";

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
    kernel_ = cl::Kernel(program, "decimateFilterY");

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
    kernel_.setArg(3, filter_);

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



void QuadToComplexDecimateFilterY::operator() (cl::CommandQueue& cq, 
                 ImageBuffer<cl_float>& input, 
                 cl::Image2D& output0,
                 cl::Image2D& output1,
                 const std::vector<cl::Event>& waitEvents,
                 cl::Event* doneEvent)
{
    // Padding etc.
    cl::NDRange workgroupSize = {workgroupSize_, workgroupSize_};
    cl::NDRange offset = {padding_, padding_};

    // Must have the padding the kernel expects
    assert(input.padding() == padding_);

    // Pad symmetrically if needed
    bool symmetricPadding = (input.height() % 4) == 2;

    // Input and output formats need to be compatible
    const size_t quadHeight = (input.height() + symmetricPadding * 2) / 2;
    assert(input.width() == 2*output0.getImageInfo<CL_IMAGE_WIDTH>());
    assert(quadHeight == 2*output0.getImageInfo<CL_IMAGE_HEIGHT>());
    assert(input.width() == 2*output1.getImageInfo<CL_IMAGE_WIDTH>());
    assert(quadHeight == 2*output1.getImageInfo<CL_IMAGE_HEIGHT>());

    cl::NDRange globalSize = {
        roundWGs(input.width(), workgroupSize[0]), 
        roundWGs(quadHeight, workgroupSize[1])
    }; 


    // Set all the arguments
    kernel_.setArg(0, input.buffer());
    kernel_.setArg(1, output0);
    kernel_.setArg(2, output1);
    kernel_.setArg(4, int(input.height()));
    kernel_.setArg(5, int(input.stride()));
    kernel_.setArg(6, int(symmetricPadding));

    // Execute
    cq.enqueueNDRangeKernel(kernel_, offset,
                            globalSize, workgroupSize,
                            &waitEvents, doneEvent);
}


