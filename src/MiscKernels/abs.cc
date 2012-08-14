#include "abs.h"
#include <iostream>
#include "util/clUtil.h"


#define STRING(t) #t

Abs::Abs(cl::Context& context, const std::vector<cl::Device>& devices)
{
    std::string src = STRING(
        __kernel void absKernel(__read_only image2d_t input,
                          __write_only image2d_t output)
        {
            sampler_t s = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST;

            int x = get_global_id(0);
            int y = get_global_id(1);

            if (x < get_image_width(output)
                && y < get_image_height(output)) {

                float2 valIn = read_imagef(input, s, (int2)(x, y)).xy;
                write_imagef(output, (int2)(x, y), 5.f * fast_length(valIn));
            }

        }
    );

    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(std::make_pair(src.c_str(), src.length()));

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
    kernel_ = cl::Kernel(program, "absKernel");
}



void Abs::operator() (cl::CommandQueue& cq, const cl::Image2D& input,
                                       const cl::Image2D& output,
                 const std::vector<cl::Event>& waitEvents,
                 cl::Event* doneEvent)
{
    const int wgSize = 16;

    cl::NDRange workgroupSize = {wgSize, wgSize};

    cl::NDRange globalSize = {
        roundWGs(output.getImageInfo<CL_IMAGE_WIDTH>(), wgSize), 
        roundWGs(output.getImageInfo<CL_IMAGE_HEIGHT>(), wgSize)
    }; 


    // Set all the arguments
    kernel_.setArg(0, input);
    kernel_.setArg(1, output);

    // Execute
    cq.enqueueNDRangeKernel(kernel_, cl::NullRange,
                            globalSize, workgroupSize,
                            &waitEvents, doneEvent);
}


