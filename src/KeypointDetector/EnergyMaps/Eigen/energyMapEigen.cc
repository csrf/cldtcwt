#include "energyMapEigen.h"
#include <cmath>

#include "util/clUtil.h"

// Specify to build everything for debug
static const char clBuildOptions[] = "";


#include <iostream>


#include "kernel.h"

EnergyMapEigen::EnergyMapEigen(cl::Context& context,
                     const std::vector<cl::Device>& devices)
   : context_(context)
{
    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(std::make_pair(
                reinterpret_cast<const char*>(EigenNS::kernel_cl), 
                EigenNS::kernel_cl_len));


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
    kernel_ = cl::Kernel(program, "energyMap");

}




void EnergyMapEigen::operator() (cl::CommandQueue& commandQueue,
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

















