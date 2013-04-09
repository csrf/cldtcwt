#include "interpPhaseMap.h"
#include <cmath>

#include "util/clUtil.h"

// Specify to build everything for debug
static const char clBuildOptions[] = "";

#include <string>
#include <sstream>
#include <iostream>
#include <iterator>
#include <algorithm>

#include <stdexcept>


#include "kernel.h"
using namespace InterpPhaseMapNS;

InterpPhaseMap::InterpPhaseMap(cl::Context& context,
                     const std::vector<cl::Device>& devices)
   : context_(context)
{
    // The OpenCL kernel:
    std::ostringstream kernelInput;

    // Define some constants
    kernelInput << "#define WG_SIZE_X (16)\n"
                   "#define WG_SIZE_Y (16)\n";
   
    // Get input from the source file
    const char* fileText = reinterpret_cast<const char*> (kernel_cl);
    size_t fileTextLength = kernel_cl_len;

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
        program.build(devices, clBuildOptions);
    } catch(cl::Error err) {
	    std::cerr 
		    << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
		    << std::endl;
	    throw;
    } 
        
    // ...and extract the useful part, viz the kernel
    kernel_ = cl::Kernel(program, "interpPhaseMap");

}




void InterpPhaseMap::operator() (cl::CommandQueue& commandQueue,
                            const Subbands& subbands,
                            cl::Image2D& energyMap,
           const std::vector<cl::Event>& preconditions,
                            cl::Event* doneEvent)
{
    // Set up all the arguments to the kernel
    kernel_.setArg(0, subbands.buffer());
    kernel_.setArg(1, cl_uint(subbands.start()));
    kernel_.setArg(2, cl_uint(subbands.pitch()));
    kernel_.setArg(3, cl_uint(subbands.stride()));
    kernel_.setArg(4, cl_uint(subbands.padding()));
    kernel_.setArg(5, cl_uint(subbands.width()));
    kernel_.setArg(6, cl_uint(subbands.height()));

    kernel_.setArg(7, energyMap);

    const size_t wgSize = 16;

    cl::NDRange globalSize = {
        roundWGs(energyMap.getImageInfo<CL_IMAGE_WIDTH>(), wgSize),
        roundWGs(energyMap.getImageInfo<CL_IMAGE_HEIGHT>(), wgSize)
    };

    // Execute
    commandQueue.enqueueNDRangeKernel(kernel_, cl::NullRange,
                                      globalSize,
                                      {wgSize, wgSize},
                                      &preconditions, doneEvent);
}

















