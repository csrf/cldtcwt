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
                            const LevelOutput& levelOutput,
                            cl::Image2D& energyMap,
                            cl::Event* doneEvent)
{
    // Set up all the arguments to the kernel
    for (int n = 0; n < levelOutput.sb.size(); ++n)
        kernel_.setArg(n, levelOutput.sb[n].buffer());

    kernel_.setArg(6, cl_uint(levelOutput.sb[0].stride()));
    kernel_.setArg(7, cl_uint(levelOutput.sb[0].padding()));
    kernel_.setArg(8, cl_uint(levelOutput.sb[0].width()));
    kernel_.setArg(9, cl_uint(levelOutput.sb[0].height()));

    kernel_.setArg(10, energyMap);

    const size_t wgSize = 16;

    cl::NDRange globalSize = {
        roundWGs(energyMap.getImageInfo<CL_IMAGE_WIDTH>(), wgSize),
        roundWGs(energyMap.getImageInfo<CL_IMAGE_HEIGHT>(), wgSize)
    };

    // Execute
    commandQueue.enqueueNDRangeKernel(kernel_, cl::NullRange,
                                      globalSize,
                                      {wgSize, wgSize},
                                      &levelOutput.done, doneEvent);
}

















