#include "findMax.h"
#include "KeypointDetector/findMaxKernel.h"

#include <string>
#include <sstream>
#include <iostream>
#include <iterator>
#include <algorithm>

#include <stdexcept>
FindMax::FindMax(cl::Context& context,
               const std::vector<cl::Device>& devices)
   : context_(context)
{
    // The OpenCL kernel:
    std::ostringstream kernelInput;

    // Define some constants
    kernelInput << "#define WG_SIZE_X (16)\n"
                   "#define WG_SIZE_Y (16)\n"
                   "#define POS_LEN (" << posLen_ << ")\n";
   
    // Get input from the source file
    const char* fileText = reinterpret_cast<const char*>
            (src_KeypointDetector_findMaxKernel_h_src);
    size_t fileTextLength = src_KeypointDetector_findMaxKernel_h_src_len;

    std::copy(fileText, fileText + fileTextLength,
              std::ostream_iterator<char>(kernelInput));

    // Convert to string
    const std::string sourceCode = kernelInput.str();

    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(std::make_pair(sourceCode.c_str(), sourceCode.length()));

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
        
    // ...and extract the useful part, i.e. the kernel
    kernel_ = cl::Kernel(program, "findMax");
}

static int roundWGs(int l, int lWG)
{
    return lWG * (l / lWG + ((l % lWG) ? 1 : 0)); 
}

void FindMax::operator() 
      (cl::CommandQueue& commandQueue,
       cl::Image& input,        float inputScale,
       cl::Image& inputFiner,   float finerScale,
       cl::Image& inputCoarser, float coarserScale,
       float threshold, float eigenRatioThreshold,
       cl::Buffer& output,
       cl::Buffer& numOutputs,
       unsigned int numOutputsOffset,
       const std::vector<cl::Event>& waitEvents,
       cl::Event* doneEvent)
{
    // Looks at the input, and finds the locations of the maxima that are at least
    // threshold, and greater than the corresponding locations in inputFiner and
    // inputCoarser.  The centres of the images are assumed to be the same.  The
    // *Scale says how many real distance units there are per pixel of input*.
    //
    // No more outputs are produced than will fit into the buffer output, and the
    // values placed there are floats in (x, y) format relative to the centre of the
    // image, in real distance units.  The total number of maxima found is placed in
    // numOutputs[numOutputsOffset] as an integer.
    //
    // The command will not start until all of waitEvents have completed, and
    // once done will flag doneEvent.

    cl::NDRange WorkgroupSize = {wgSizeX_, wgSizeY_};

    cl::NDRange GlobalSize = {
        roundWGs(input.getImageInfo<CL_IMAGE_WIDTH>(), wgSizeX_), 
        roundWGs(input.getImageInfo<CL_IMAGE_HEIGHT>(), wgSizeY_)
    }; 

    // Set all the arguments
    kernel_.setArg(0, sizeof(input), &input);
    kernel_.setArg(1, (inputScale));
    kernel_.setArg(2, sizeof(inputFiner), &inputFiner);
    kernel_.setArg(3, (finerScale));
    kernel_.setArg(4, sizeof(inputCoarser), &inputCoarser);
    kernel_.setArg(5, (coarserScale));
    kernel_.setArg(6, (threshold));
    kernel_.setArg(7, (eigenRatioThreshold));
    kernel_.setArg(8, output);
    kernel_.setArg(9, numOutputs);
    kernel_.setArg(10, (numOutputsOffset));
    kernel_.setArg(11, int(output.getInfo<CL_MEM_SIZE>() 
                            / (posLen_ * sizeof(float)))); // Max number of outputs

    // Execute
    commandQueue.enqueueNDRangeKernel(kernel_, cl::NullRange,
                                      GlobalSize,
                                      WorkgroupSize,
                                      &waitEvents, doneEvent);
}


size_t FindMax::getPosLength() const
{
    return posLen_;
}

