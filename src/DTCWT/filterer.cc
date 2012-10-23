#include "filterer.h"
#include "util/clUtil.h"
#include <stdexcept>
#include <iostream>
#include <string>
#include <sstream>

// Specify to build everything for debug
static const char clBuildOptions[] = "";

// Central sampler creating function (to make changing the addressing
// overflow behaviour easy, for upgrade to OpenCL 1.1)
cl::Sampler createSampler(cl::Context& context)
{
    return cl::Sampler(context, CL_FALSE, CL_ADDRESS_CLAMP,
                       CL_FILTER_NEAREST);
}



static const std::string reflectRepeat = "CLK_ADDRESS_NONE";
//MIRRORED_REPEAT";
//"CLK_ADDRESS_CLAMP";



Filter::Filter(cl::Context& context,
               const std::vector<cl::Device>& devices,
               cl::Buffer coefficients,
               Direction dimension)
   : context_(context), coefficients_(coefficients), dimension_(dimension),
     wgSizeX_(16), wgSizeY_(16)
{
    // The OpenCL kernel:
    std::ostringstream kernelInput;

    // Filter must be odd-lengthed

    // Need to work out the filter length; if this value is passed directly,
    // the setArg function doesn't understand its type properly.
    const int filterLength = coefficients_.getInfo<CL_MEM_SIZE>() 
                                / sizeof(float);

    const int offset = (filterLength-1) / 2;

    const int inputLocalSizeX = wgSizeX_
                          + ((dimension_ == x) ? (filterLength - 1) : 0);
    const int inputLocalSizeY = wgSizeY_
                          + ((dimension_ == y) ? (filterLength - 1) : 0);

    kernelInput
    << "__kernel void filter(__read_only image2d_t input,           \n"
        "                        __write_only image2d_t output,         \n"
        "                        __constant float* filter)          \n"
        "{                                                              \n"
            "sampler_t inputSampler ="
                "CLK_NORMALIZED_COORDS_FALSE"
                "| " << reflectRepeat <<
                "| CLK_FILTER_NEAREST;"

            "__local float inputLocal[" << inputLocalSizeY << "]"
                                    "[" << inputLocalSizeX << "];"

            "const int filterLength = " << filterLength << ";"

            "const int gx = get_global_id(0),"
                      "gy = get_global_id(1),"
                      "lx = get_local_id(0),"
                      "ly = get_local_id(1);\n";

    if (dimension_ == y) {

        kernelInput << 
            // Load the local store

            "int startY = get_local_size(1) * get_group_id(1)"
                            "- " << offset << ";"

            "for (int n = 0;"
                 "(n * " << wgSizeY_ << ") < " << inputLocalSizeY << ";"
                 "++n) {"

                 // Make sure still in range, then read
                 "if ((ly+n*" << wgSizeY_ << ") < " << inputLocalSizeY << ") {"
                    "int readY = startY+ly+n*" << wgSizeY_ << ";"
                    "int height = get_image_height(input);"

                    // Deal with extension correctly
                    "readY = (readY < 0) ? (-readY - 1): readY;"
                    "readY = (readY >= height)?"
                            "height - (readY - height) - 1: readY;"
                                
                    "inputLocal[ly + n * " << wgSizeY_ << "][lx]"
                        "= read_imagef(input, inputSampler,"
                          "(float2) (gx, readY)).x;"
                          // If the position is less than 0, we need to add
                          // an offset of one so that the reflected
                          // position is correct
                  "}"

            "}"

            "barrier(CLK_LOCAL_MEM_FENCE);"

            // Do the filtering
            "float out = 0.0f;"
            "for (int i = 0; i < filterLength; ++i)"
                 "out += filter[filterLength-1-i] *"
                         "inputLocal[ly + i][lx];";

    } else if (dimension_ == x) {

        kernelInput << 
            // Load the local store

            "int startX = get_local_size(0) * get_group_id(0)"
                            "- " << offset << ";"

            "for (int n = 0;"
                 "(n * " << wgSizeX_ << ") < " << inputLocalSizeX << ";"
                 "++n) {"

                "int readX = startX+lx+n*" << wgSizeX_ << ";"
                "int width = get_image_width(input);"

                // Deal with extension correctly
                "readX = (readX < 0) ? (-readX - 1): readX;"
                "readX = (readX >= width)?"
                        "width - (readX - width) - 1: readX;"
 
                 // Make sure still in range, then read
                 "if ((lx+n*" << wgSizeX_ << ") < " << inputLocalSizeX << ")"
                    "inputLocal[ly][lx + n * " << wgSizeX_ << "]"
                        "= read_imagef(input, inputSampler,"
                          "(int2) (readX, gy)).x;"

            "}"

            "barrier(CLK_LOCAL_MEM_FENCE);"

            // Do the filtering
            "float out = 0.0f;"
            "for (int i = 0; i < filterLength; ++i)"
                 "out += filter[filterLength-1-i] *"
                         "inputLocal[ly][lx+i];";  
 
    }
            
    kernelInput <<
            // Write the result
            "if (gx < get_image_width(output)"
             "&& gy < get_image_height(output))"
                "write_imagef(output, (int2) (gx, gy), out);"
        "}";

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
    kernel_ = cl::Kernel(program, "filter");

    // We know what the filter argument will be already
    kernel_.setArg(2, coefficients_);
}



void Filter::operator() 
      (cl::CommandQueue& commandQueue,
       cl::Image& input,
       cl::Image& output,
       const std::vector<cl::Event>& waitEvents,
       cl::Event* doneEvent)
{
    // Run the filter for each location in output (which determines
    // the locations to run at) using commandQueue.  input and output are
    // both single-component float images.  filter is a vector of floats.
    // The command will not start until all of waitEvents have completed, and
    // once done will flag doneEvent.

    cl::NDRange WorkgroupSize = {wgSizeX_, wgSizeY_};

    cl::NDRange GlobalSize = {
        roundWGs(output.getImageInfo<CL_IMAGE_WIDTH>(), wgSizeX_), 
        roundWGs(output.getImageInfo<CL_IMAGE_HEIGHT>(), wgSizeY_)
    }; 

    // Set all the arguments
    kernel_.setArg(0, sizeof(input), &input);
    kernel_.setArg(1, sizeof(output), &output);

    // Execute
    commandQueue.enqueueNDRangeKernel(kernel_, cl::NullRange,
                                      GlobalSize,
                                      WorkgroupSize,
                                      &waitEvents, doneEvent);
}








DecimateFilter::DecimateFilter(cl::Context& context,
               const std::vector<cl::Device>& devices,
               cl::Buffer coefficients,
               Direction dimension, bool swapTrees)
   : context_(context), coefficients_(coefficients), dimension_(dimension),
     wgSizeX_(16), wgSizeY_(16)
{
    // The OpenCL kernel:
    std::ostringstream kernelInput;

    // Filter must be odd-lengthed

    // Need to work out the filter length; if this value is passed directly,
    // the setArg function doesn't understand its type properly.
    const int filterLength = coefficients_.getInfo<CL_MEM_SIZE>() 
                                / sizeof(float);

    const int inputLocalSizeX = 
        (dimension_ == x)? 4 * (wgSizeX_ - 1) + 2 * filterLength
                         : wgSizeX_;
    const int inputLocalSizeY = 
        (dimension_ == y)? 4 * (wgSizeY_ - 1) + 2 * filterLength
                         : wgSizeY_;


    kernelInput
    << "__kernel void decimateFilter(__read_only image2d_t input,"
                                    "__write_only image2d_t output,"
                                    "__constant float* filter,"
                                    "int offset)"
        "{"
            "sampler_t inputSampler ="
                "CLK_NORMALIZED_COORDS_FALSE"
                "| " << reflectRepeat <<
                "| CLK_FILTER_NEAREST;"

            "__local float inputLocal[" << inputLocalSizeY << "]"
                                    "[" << inputLocalSizeX << "];"

            "const int filterLength = " << filterLength << ";"

            "const int gx = get_global_id(0),"
                      "gy = get_global_id(1),"
                      "lx = get_local_id(0),"
                      "ly = get_local_id(1);";

    if (dimension_ == y) {

        kernelInput << 
            
            // Load the local store

            // Find the start of the group
            "int y = get_local_size(1) * get_group_id(1);"
            "int startY = 4 * y - (" << filterLength << "-2) + offset;"

            "for (int n = 0;"
                 "(n * " << wgSizeY_ << ") < " << inputLocalSizeY << ";"
                 "++n) {"
                 "if ((ly + n * " << wgSizeY_ << ") < " << inputLocalSizeY << "){"
                     "int readY = startY+ly+n*" << wgSizeY_ << ";"
                     "int height = get_image_height(input);"

                     // Deal with extension correctly
                     "readY = (readY < 0) ? (-readY - 1): readY;"
                     "readY = (readY >= height)?"
                              "height - (readY - height) - 1: readY;"
             

                    "inputLocal[ly + n * " << wgSizeY_ << "][lx]"
                        "= read_imagef(input, inputSampler,"
                          "(int2) (gx, readY)).x;"

                "}"
            "}"               

            "barrier(CLK_LOCAL_MEM_FENCE);"

            "if (gx < get_image_width(output)"
             "&& (2*gy) < get_image_height(output)) {"

                // Do the filtering: start from 4*ly (due to
                // decimation/interleaved trees) and select 2*i(+1) due to the
                // interleaved trees
                "float out1 = 0.0f, out2 = 0.0f;"
                "for (int i = 0; i < filterLength; ++i) {"
                     "out1 += filter[filterLength-1-i]"
                              "* inputLocal[2*(2*ly + i)][lx];"
                     "out2 += filter[i]"
                              "* inputLocal[2*(2*ly + i) + 1][lx];"
                 "}";

                // Write the result
        if (!swapTrees)
            kernelInput << 
                "write_imagef(output, (int2) (gx, 2*gy), out1);"
                "write_imagef(output, (int2) (gx, 2*gy+1), out2);";
        else
            kernelInput << 
                "write_imagef(output, (int2) (gx, 2*gy+1), out1);"
                "write_imagef(output, (int2) (gx, 2*gy), out2);";

        kernelInput << 
            "}"
        "}";

    } else if (dimension_ == x) {

        kernelInput << 
            
            // Load the local store

            // Find the start of the group
            "int x = get_local_size(0) * get_group_id(0);"
            "int startX = 4 * x - (" << filterLength << "-2) + offset;"

            "for (int n = 0;"
                 "(n * " << wgSizeX_ << ") < " << inputLocalSizeX << ";"
                 "++n) {"
                 "if ((lx + n * " << wgSizeX_ << ") < " << inputLocalSizeX << ") {"
                    "int readX = startX+lx+n*" << wgSizeX_ << ";"
                    "int width = get_image_width(input);"

                    // Deal with extension correctly
                    "readX = (readX < 0) ? (-readX - 1): readX;"
                    "readX = (readX >= width)?"
                            "width - (readX - width) - 1: readX;"
 
                    "inputLocal[ly][lx + n * " << wgSizeX_ << "]"
                        "= read_imagef(input, inputSampler,"
                          "(int2) (readX, gy)).x;"
                "}"
            "}"               

            "barrier(CLK_LOCAL_MEM_FENCE);"

            "if ((2*gx) < get_image_width(output)"
             "&& gy < get_image_height(output)) {"

                // Do the filtering: start from 4*lx (due to
                // decimation/interleaved trees) and select 2*i(+1) due to the
                // interleaved trees
                "float out1 = 0.0f, out2 = 0.0f;"
                "for (int i = 0; i < filterLength; ++i) {"
                     "out1 += filter[filterLength-1-i]"
                              "* inputLocal[ly][2*(2*lx + i)];"
                     "out2 += filter[i]"
                              "* inputLocal[ly][2*(2*lx + i) + 1];"
                 "}";

                // Write the result
        if (!swapTrees)
            kernelInput << 
                "write_imagef(output, (int2) (2*gx, gy), out1);"
                "write_imagef(output, (int2) (2*gx+1, gy), out2);";
        else
            kernelInput << 
                "write_imagef(output, (int2) (2*gx+1, gy), out1);"
                "write_imagef(output, (int2) (2*gx, gy), out2);";


        kernelInput << 
            "}"
        "}";
    }
            

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
    kernel_ = cl::Kernel(program, "decimateFilter");

    // We know what the filter argument will be already
    kernel_.setArg(2, coefficients_);
}



void DecimateFilter::operator() (cl::CommandQueue& commandQueue,
               const cl::Image2D& input,
               cl::Image2D& output,
               const std::vector<cl::Event>& waitEvents,
               cl::Event* doneEvent)
{
    // Run the decimation filter for each location in output (which
    // determines the locations to run at) using commandQueue.  input and 
    // output are both single-component float images.  filter is a vector of
    // floats. The command will not start until all of waitEvents have 
    // completed, and once done will flag doneEvent.


    const int width = output.getImageInfo<CL_IMAGE_WIDTH>(),
              height = output.getImageInfo<CL_IMAGE_HEIGHT>();

    cl::NDRange GlobalSize = {
        roundWGs(width / (dimension_ == x? 2 : 1), wgSizeX_), 
        roundWGs(height / (dimension_ == y? 2 : 1), wgSizeY_)
    }; 

    // Make sure the resulting image is an even height, i.e. it has the
    // same length for both the trees
    bool pad = (((dimension_ == x)? width : height) % 4) != 0;

    // Tell the kernel to use the buffers, and how long they are
    kernel_.setArg(0, input);         // input
    kernel_.setArg(1, output);        // output
    kernel_.setArg(3, pad? -1 : 0);   // Offset to start reading input from

    // Execute
    commandQueue.enqueueNDRangeKernel(kernel_, cl::NullRange,
                                      GlobalSize,
                                      {wgSizeX_, wgSizeY_},
                                      &waitEvents, doneEvent);

}








void cornernessMap(cl::Context& context,
                   cl::CommandQueue& commandQueue,
                   cl::Kernel& cornernessMapKernel,
                   cl::Image2D& output, 
                   std::vector<cl::Image2D> subbands)
{
    // The output is the same size as each of the inputs
    const int width = subbands[0].getImageInfo<CL_IMAGE_WIDTH>();
    const int height = subbands[0].getImageInfo<CL_IMAGE_HEIGHT>();
    output = createImage2D(context, width, height);

    // Send across the real and imaginary components
    cornernessMapKernel.setArg(0,  subbands[0]);
    cornernessMapKernel.setArg(1,  subbands[0+6]);
    cornernessMapKernel.setArg(2,  subbands[1]);
    cornernessMapKernel.setArg(3,  subbands[1+6]);
    cornernessMapKernel.setArg(4,  subbands[2]);
    cornernessMapKernel.setArg(5,  subbands[2+6]);
    cornernessMapKernel.setArg(6,  subbands[3]);
    cornernessMapKernel.setArg(7,  subbands[3+6]);
    cornernessMapKernel.setArg(8,  subbands[4]);
    cornernessMapKernel.setArg(9,  subbands[4+6]);
    cornernessMapKernel.setArg(10, subbands[5]);
    cornernessMapKernel.setArg(11, subbands[5+6]);

    cornernessMapKernel.setArg(12, createSampler(context));

    cornernessMapKernel.setArg(13, output);

    // Execute
    commandQueue.enqueueNDRangeKernel(cornernessMapKernel, cl::NullRange,
                                      cl::NDRange(width, height),
                                      cl::NullRange);
    commandQueue.finish();
}




QuadToComplex::QuadToComplex(cl::Context& context_,
                     const std::vector<cl::Device>& devices)
   : context(context_)
{
    // The OpenCL kernel:
    const std::string sourceCode = 
        "__kernel void quadToComplex(__read_only image2d_t input,"
                                    "sampler_t inputSampler,"
                                    "__write_only image2d_t out1,"
                                    "__write_only image2d_t out2)"
        "{"
            "int x = get_global_id(0);"
            "int y = get_global_id(1);"

            "if (x < get_image_width(out1)"
             "&& y < get_image_height(out1)) {"

                // Sample upper left, upper right, etc
                "float ul = read_imagef(input, inputSampler,"
                                      "(int2) (  2*x,   2*y)).x;"
                "float ur = read_imagef(input, inputSampler,"
                                      "(int2) (2*x+1,   2*y)).x;"
                "float ll = read_imagef(input, inputSampler,"
                                      "(int2) (  2*x, 2*y+1)).x;"
                "float lr = read_imagef(input, inputSampler,"
                                      "(int2) (2*x+1, 2*y+1)).x;"

                "const float factor = 1.0f / sqrt(2.0f);"

                // Combine into complex pairs
                "write_imagef(out1, (int2) (x, y),"
                             "factor * (float4) (ul - lr, ur + ll, 0.0, 1.0));"
                "write_imagef(out2, (int2) (x, y),"
                             "factor * (float4) (ul + lr, ur - ll, 0.0, 1.0));"

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
    kernel = cl::Kernel(program, "quadToComplex");

    // The sampler, later to be used as a kernel argument
    sampler = createSampler(context);

    kernel.setArg(1, sampler);
}



void QuadToComplex::operator() (cl::CommandQueue& commandQueue,
               const cl::Image2D& input,
               cl::Image2D& out1, cl::Image2D& out2,
               const std::vector<cl::Event>& waitEvents,
               cl::Event* doneEvent)
{
    // Set up all the arguments to the kernel
    kernel.setArg(0, input);
    kernel.setArg(2, out1);
    kernel.setArg(3, out2);

    const size_t wgSize = 16;

    cl::NDRange globalSize = {
        roundWGs(out1.getImageInfo<CL_IMAGE_WIDTH>(), wgSize),
        roundWGs(out1.getImageInfo<CL_IMAGE_HEIGHT>(), wgSize)
    };

    // Execute
    commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                      globalSize,
                                      {wgSize, wgSize},
                                      &waitEvents, doneEvent);
}




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
        program.build(devices, clBuildOptions);
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



