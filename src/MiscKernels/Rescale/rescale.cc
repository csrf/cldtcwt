#include "rescale.h"
#include "util/clUtil.h"

#include <iostream>

Rescale::Rescale(cl::Context& context,
                 const std::vector<cl::Device>& devices)
    : context_(context)
{
    const std::string sourceCode = 
    "__kernel void rescale(__read_only image2d_t input,"
                          "__write_only image2d_t output,"
                          "const float scaleFactor)\n"
    "{\n"
        "int x = get_global_id(0);\n"
        "int y = get_global_id(1);\n"
        
        // Make sure there is work to be done before starting it
        "if (x < get_image_width(output)"
         "&& y < get_image_height(output)) {\n"

            "sampler_t inputSampler ="
                "CLK_NORMALIZED_COORDS_FALSE"
                "| CLK_ADDRESS_MIRRORED_REPEAT"
                "| CLK_FILTER_LINEAR;\n"


            // The coordinates are scaled about the centre of the image,
            // so find where the centre is
            "int2 outCoords = (int2) (x, y);\n"

            "float2 outCoordsRelCentre = (float2) (x, y) - "
                "((float2) (get_image_width(output),"
                           "get_image_height(output)) - 1.0f) / 2.0f;\n"


            // Work out the input coordinates relative to its centre, then
            // work back from there to find the (x,y) from the corner
            "float2 inCoords = outCoordsRelCentre / scaleFactor"
                "+ ((float2) (get_image_width(input),"
                             "get_image_height(input)) - 1.0f) /  2.0f;\n"


            // Read and write
            "float val = read_imagef(input, inputSampler, inCoords).x;\n"
            "write_imagef(output, outCoords, val);\n"
        "}"
    "}";

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
        
    // ...and extract the useful part, viz the kernel
    kernel_ = cl::Kernel(program, "rescale");
}





void Rescale::operator() 
      (cl::CommandQueue& commandQueue,
       cl::Image& input,
       cl::Image2D& output,
       float scalingFactor,
       const std::vector<cl::Event>& waitEvents,
       cl::Event* doneEvent)
{
    const int wgSizeX_ = 16, wgSizeY_ = 16;

    // Break the task up into blocks of 16 to ensure workgroups operate on
    // pixels close together
    cl::NDRange WorkgroupSize = {wgSizeX_, wgSizeY_};

    // Make big enough to include all pixels (at least)
    cl::NDRange GlobalSize = {
        roundWGs(output.getImageInfo<CL_IMAGE_WIDTH>(), wgSizeX_), 
        roundWGs(output.getImageInfo<CL_IMAGE_HEIGHT>(), wgSizeY_)
    }; 

    // Set all the arguments
    kernel_.setArg(0, sizeof(input), &input);
    kernel_.setArg(1, output);
    kernel_.setArg(2, float(scalingFactor));


    // Execute
    commandQueue.enqueueNDRangeKernel(kernel_, cl::NullRange,
                                      GlobalSize,
                                      WorkgroupSize,
                                      &waitEvents, doneEvent);
}


