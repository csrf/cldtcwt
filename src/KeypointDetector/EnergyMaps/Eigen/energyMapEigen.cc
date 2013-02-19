#include "energyMapEigen.h"
#include <cmath>

#include "util/clUtil.h"

// Specify to build everything for debug
static const char clBuildOptions[] = "";


#include <iostream>



EnergyMapEigen::EnergyMapEigen(cl::Context& context,
                     const std::vector<cl::Device>& devices)
   : context_(context)
{
    // The OpenCL kernel:
    const std::string sourceCode = 
        "__kernel void energyMap(__read_only image2d_t sb0,"
                                "__read_only image2d_t sb1,"
                                "__read_only image2d_t sb2,"
                                "__read_only image2d_t sb3,"
                                "__read_only image2d_t sb4,"
                                "__read_only image2d_t sb5,"
                                "__write_only image2d_t out)"
        "{"
            "sampler_t s ="
                "CLK_NORMALIZED_COORDS_FALSE"
                "| CLK_ADDRESS_CLAMP"
                "| CLK_FILTER_NEAREST;"

            "int x = get_global_id(0);"
            "int y = get_global_id(1);"

            "if (x < get_image_width(out)"
             "&& y < get_image_height(out)) {"

                // Sample each subband
                "float2 h0 = read_imagef(sb0, s, (int2) (x,y)).s01;"
                "float2 h1 = read_imagef(sb1, s, (int2) (x,y)).s01;"
                "float2 h2 = read_imagef(sb2, s, (int2) (x,y)).s01;"
                "float2 h3 = read_imagef(sb3, s, (int2) (x,y)).s01;"
                "float2 h4 = read_imagef(sb4, s, (int2) (x,y)).s01;"
                "float2 h5 = read_imagef(sb5, s, (int2) (x,y)).s01;"

                // Convert to absolute (still squared, because it's more
                // convenient)
                "float abs_h_2[] = { h0.s0 * h0.s0 + h0.s1 * h0.s1,"
                "                    h1.s0 * h1.s0 + h1.s1 * h1.s1,"
                "                    h2.s0 * h2.s0 + h2.s1 * h2.s1,"
                "                    h3.s0 * h3.s0 + h3.s1 * h3.s1,"
                "                    h4.s0 * h4.s0 + h4.s1 * h4.s1,"
                "                    h5.s0 * h5.s0 + h5.s1 * h5.s1};"
                
                // Approximate angular frequencies
                "float wx[] = {-1.4612, -3.2674, -4.3836, -4.3836, -3.2674, -1.461};"
                "float wy[] = {-4.3836, -3.2674, -1.4612,  1.4612,  3.2674,  4.3836};"

                "float H00 = 0, H11 = 0, H01 = 0;"

                "for (int n = 0; n < 6; ++n) {"
                    "H00 -= wx[n] * wx[n] * abs_h_2[n];"
                    "H11 -= wy[n] * wy[n] * abs_h_2[n];"
                    "H01 -= wx[n] * wy[n] * abs_h_2[n];"
                "}"

                "float root = sqrt(H00 * H00 + H11 * H11 - 2.f * H11 * H00"
                                  "+ 4.f * H01 * H01);"

                "float l0 = -(H00 + H11 + root) / 2.f;"
                "float l1 = -(H00 + H11 - root) / 2.f;"


                // Calculate result
                "float result = l0 / (l1 + 1000.f);" 

                // Produce output
                "write_imagef(out, (int2) (x, y), result);"

            "}"

        "}";

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
    kernel_ = cl::Kernel(program, "energyMap");

}




void EnergyMapEigen::operator() (cl::CommandQueue& commandQueue,
                            const LevelOutput& levelOutput,
                            cl::Image2D& energyMap,
                            cl::Event* doneEvent)
{
    // Set up all the arguments to the kernel
    for (int n = 0; n < levelOutput.sb.size(); ++n)
        kernel_.setArg(n, levelOutput.sb[n]);

    kernel_.setArg(levelOutput.sb.size(), energyMap);

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

















