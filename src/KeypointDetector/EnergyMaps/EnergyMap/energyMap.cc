#include "energyMap.h"
#include "util/clUtil.h"

#include <iostream>

#include "kernel.h"

EnergyMap::EnergyMap(cl::Context& context,
                     const std::vector<cl::Device>& devices)
   : context_(context)
{
    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(std::make_pair(
                reinterpret_cast<const char*>(EnergyMapNS::kernel_cl), 
                EnergyMapNS::kernel_cl_len));

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
    kernel_ = cl::Kernel(program, "energyMap");

}




void EnergyMap::operator() (cl::CommandQueue& commandQueue,
                            const LevelOutput& levelOutput,
                            cl::Image2D& energyMap,
                            cl::Event* doneEvent)
{
    // Set up all the arguments to the kernel
    for (int n = 0; n < levelOutput.sb.size(); ++n)
        kernel_.setArg(n, levelOutput.sb[n].buffer());

    int n = 6;
    kernel_.setArg(n++, cl_uint(levelOutput.sb[0].stride()));
    kernel_.setArg(n++, cl_uint(levelOutput.sb[0].padding()));
    kernel_.setArg(n++, cl_uint(levelOutput.sb[0].width()));
    kernel_.setArg(n++, cl_uint(levelOutput.sb[0].height()));

    kernel_.setArg(n++, energyMap);

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



