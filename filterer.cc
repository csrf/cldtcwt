#include "filterer.h"
#include "clUtil.h"
#include <stdexcept>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

// Central sampler creating function (to make changing the addressing
// overflow behaviour easy, for upgrade to OpenCL 1.1)
cl::Sampler createSampler(cl::Context& context)
{
    return cl::Sampler(context, CL_FALSE, CL_ADDRESS_CLAMP,
                       CL_FILTER_NEAREST);
}



static const std::string reflectRepeat = "CLK_ADDRESS_MIRRORED_REPEAT";
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
                 "if ((ly+n*" << wgSizeY_ << ") < " << inputLocalSizeY << ")"
                    "inputLocal[ly + n * " << wgSizeY_ << "][lx]"
                        "= read_imagef(input, inputSampler,"
                          "(int2) (gx, startY+ly+n*" << wgSizeY_ << ")).x;"

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

                 // Make sure still in range, then read
                 "if ((lx+n*" << wgSizeX_ << ") < " << inputLocalSizeX << ")"
                    "inputLocal[ly][lx + n * " << wgSizeX_ << "]"
                        "= read_imagef(input, inputSampler,"
                          "(int2) (startX+lx+n*" << wgSizeX_ << ", gy)).x;"

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
        program.build(devices);
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

static int roundWGs(int l, int lWG)
{
    return lWG * (l / lWG + ((l % lWG) ? 1 : 0)); 
}

void Filter::operator() 
      (cl::CommandQueue& commandQueue,
       const cl::Image2D& input,
       cl::Image2D& output,
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
    kernel_.setArg(0, input);
    kernel_.setArg(1, output);

    // Execute
    commandQueue.enqueueNDRangeKernel(kernel_, cl::NullRange,
                                      GlobalSize,
                                      WorkgroupSize,
                                      &waitEvents, doneEvent);
}


cl::Image2D Filter::dummyRun(const cl::Image2D& input)
{
    return dummyRun(input.getImageInfo<CL_IMAGE_WIDTH>(),
                    input.getImageInfo<CL_IMAGE_HEIGHT>());
}


cl::Image2D Filter::dummyRun(size_t inWidth, size_t inHeight)
{
    if (dimension_ == x)
        return createImage2D(context_, inWidth + inWidth % 2, inHeight);
    else
        return createImage2D(context_, inWidth, inHeight + inHeight % 2);
}










DecimateFilter::DecimateFilter(cl::Context& context,
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
                 "if ((ly + n * " << wgSizeY_ << ") < " << inputLocalSizeY << ")"
                    "inputLocal[ly + n * " << wgSizeY_ << "][lx]"
                        "= read_imagef(input, inputSampler,"
                          "(int2) (gx, startY + ly + n * " << wgSizeY_ << ")).x;"
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
                 "}"

                // Write the result
                "write_imagef(output, (int2) (gx, 2*gy), out1);"
                "write_imagef(output, (int2) (gx, 2*gy+1), out2);"

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
                 "if ((lx + n * " << wgSizeX_ << ") < " << inputLocalSizeX << ")"
                    "inputLocal[ly][lx + n * " << wgSizeX_ << "]"
                        "= read_imagef(input, inputSampler,"
                          "(int2) (startX + lx + n * " << wgSizeX_ << ", gy)).x;"
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
                 "}"

                // Write the result
                "write_imagef(output, (int2) (2*gx, gy), out1);"
                "write_imagef(output, (int2) (2*gx+1, gy), out2);"

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
        program.build(devices);
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




cl::Image2D DecimateFilter::dummyRun(const cl::Image2D& input)
{
    return dummyRun(input.getImageInfo<CL_IMAGE_WIDTH>(),
                    input.getImageInfo<CL_IMAGE_HEIGHT>());
}



cl::Image2D DecimateFilter::dummyRun(size_t inWidth, size_t inHeight)
{
    if (dimension_ == x) {
        bool pad = (inWidth % 4) != 0;
        return createImage2D(context_, inWidth / 2 + (pad? 1 : 0), inHeight);
    } else {
        bool pad = (inHeight % 4) != 0;
        return createImage2D(context_, inWidth, inHeight / 2 + (pad? 1 : 0));
    }
}





ColFilter::ColFilter(cl::Context& context_,
                     const std::vector<cl::Device>& devices)
   : context(context_)
{
    // The OpenCL kernel:
    std::ostringstream kernelInput;

    kernelInput
    << "__kernel void colFilter(__read_only image2d_t input,           \n"
        "                        __constant float* filter,          \n"
        "                        __local float* inputLocal,      \n"
        "                        const int filterLength,                \n"
        "                        __write_only image2d_t output)         \n"
        "{                                                              \n"
        "sampler_t inputSampler ="
            "CLK_NORMALIZED_COORDS_FALSE"
            "| CLK_ADDRESS_MIRRORED_REPEAT"
            "| CLK_FILTER_NEAREST;"

        "    // Row wise filter.  filter must be odd-lengthed           \n"
        "    // Coordinates in output frame                             \n"
        "    int x = get_global_id(0);                                  \n"
        "    int y = get_global_id(1);                                  \n"
        "                                                               \n"
        "    const int offset = (filterLength-1) / 2;                 \n"
        "    const int l = get_local_id(1);                             \n"
        "                                                               \n"
        "    if (l >= (get_local_size(1) - offset))                    \n"
        "       inputLocal[l - (get_local_size(1) - offset)]             \n"
        "          = read_imagef(input, inputSampler,                   \n"
        "                        (int2) (x, y - get_local_size(1))).x; \n"
        "                                                               \n"
        "    inputLocal[l]             \n"
        "       = read_imagef(input, inputSampler, (int2) (x, y)).x; \n"
        "                                                               \n"
        "    if (l < (filterLength - offset))                    \n"
        "       inputLocal[l + get_local_size(1) + offset]             \n"
        "          = read_imagef(input, inputSampler,                   \n"
        "                        (int2) (x, y + get_local_size(1))).x; \n"
        "                                                               \n"
    	"    barrier(CLK_LOCAL_MEM_FENCE);				                \n"
        "							                                	\n"
        "    // Results for each of the two trees                       \n"
        "    float out = 0.0f;                                          \n"
        "                                                               \n"
        "    // Apply the filter forward                                \n"
        "    for (int i = 0; i < filterLength; ++i)                     \n"
        "        out += filter[filterLength-1-i] *                 \n"
        "                inputLocal[get_group_id(1) + i];               \n"  
        "                                                               \n"
        "    // Output position is r rows down, plus 2*c along (because \n"
        "    // the outputs from two trees are interleaved)             \n"
        "    if (y < get_image_height(output))                         \n"
        "    write_imagef(output, (int2) (x, y), out);                  \n"
        "}                                                              \n"
        "\n";

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
        
    // ...and extract the useful part, viz the kernel
    kernel = cl::Kernel(program, "colFilter");

    // The sampler, later to be used as a kernel argument
    sampler = createSampler(context);
}


cl::Image2D ColFilter::dummyRun(const cl::Image2D& input)
{
    return dummyRun(input.getImageInfo<CL_IMAGE_WIDTH>(),
                    input.getImageInfo<CL_IMAGE_HEIGHT>());
}

cl::Image2D ColFilter::dummyRun(size_t inWidth, size_t inHeight)
{
    return createImage2D(context, inWidth, inHeight + inHeight % 2);
}



cl::Image2D ColFilter::operator() (cl::CommandQueue& commandQueue,
               cl::Image2D& input, cl::Buffer& filter,
               const std::vector<cl::Event>& waitEvents,
               cl::Event* doneEvent,
               cl::Image2D* targetImage)
{
    // Run the column filter for each location in output (which determines
    // the locations to run at) using commandQueue.  input and output are
    // both single-component float images.  filter is a vector of floats.
    // The command will not start until all of waitEvents have completed, and
    // once done will flag doneEvent.

    // Use the pre-allocated output, if given; otherwise, create a new
    // output of appropriate size
    cl::Image2D output
        = (targetImage != nullptr)? 
              *targetImage : dummyRun(input);

    const int width = output.getImageInfo<CL_IMAGE_WIDTH>(),
              height = output.getImageInfo<CL_IMAGE_HEIGHT>();
 

    // Need to work out the filter length; if this value is passed directly,
    // the setArg function doesn't understand its type properly.
    const int filterLength = filter.getInfo<CL_MEM_SIZE>() / sizeof(float);

    int heightWG = 64;
    cl::NDRange WorkgroupSize = {1, heightWG};
    int numVertWGs = height / heightWG + ((height % heightWG) ? 1 : 0); 
    cl::NDRange GlobalSize = {width, heightWG * numVertWGs }; 

    // Set all the arguments
    kernel.setArg(0, input);
    kernel.setArg(1, filter);
    kernel.setArg(2, cl::__local(sizeof(float) * (filterLength + heightWG)));
    kernel.setArg(3, filterLength);
    kernel.setArg(4, output);

    // Execute
    commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                      GlobalSize,
                                      WorkgroupSize,
                                      &waitEvents, doneEvent);

    return output;
}






RowFilter::RowFilter(cl::Context& context_,
                     const std::vector<cl::Device>& devices)
   : context(context_)
{
    // The OpenCL kernel:
    const std::string sourceCode = 
        "__kernel void rowFilter(__read_only image2d_t input,           \n"
        "                        sampler_t inputSampler,                \n"
        "                        __global const float* filter,          \n"
        "                        const int filterLength,                \n"
        "                        __write_only image2d_t output)         \n"
        "{                                                              \n"
        "    // Row wise filter.  filter must be odd-lengthed           \n"
        "    // Coordinates in output frame                             \n"
        "    int x = get_global_id(0);                                  \n"
        "    int y = get_global_id(1);                                  \n"
        "                                                               \n"
        "     // Results for each of the two trees                      \n"
        "    float out = 0.0f;                                          \n"
        "                                                               \n"
        "    // Apply the filter forward                                \n"
        "    int startX = x - (filterLength-1) / 2;                     \n"
        "    for (int i = 0; i < filterLength; ++i)                     \n"
        "        out += filter[filterLength-1-i] *                      \n"
        "                read_imagef(input, inputSampler,               \n"
        "                            (int2) (startX + i, y)).x;         \n"
        "                                                               \n"
        "    // Output position is r rows down, plus c along (because   \n"
        "    // the outputs from two trees are interleaved)             \n"
        "    write_imagef(output, (int2) (x, y), out);                  \n"
        "}                                                              \n"
        "\n";

    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(std::make_pair(sourceCode.c_str(), sourceCode.length()));

    // Compile it...
    cl::Program program(context, source);
    program.build(devices);
        
    // ...and extract the useful part, viz the kernel
    kernel = cl::Kernel(program, "rowFilter");

    // The sampler, later to be used as a kernel argument
    sampler = createSampler(context);
}


cl::Image2D RowFilter::dummyRun(const cl::Image2D& input)
{
    return dummyRun(input.getImageInfo<CL_IMAGE_WIDTH>(),
                    input.getImageInfo<CL_IMAGE_HEIGHT>());
}

cl::Image2D RowFilter::dummyRun(size_t inWidth, size_t inHeight)
{
    return createImage2D(context, inWidth + inWidth % 2, inHeight);
}




cl::Image2D RowFilter::operator() (cl::CommandQueue& commandQueue,
               cl::Image2D& input, cl::Buffer& filter,
               const std::vector<cl::Event>& waitEvents,
               cl::Event* doneEvent,
               cl::Image2D* targetImage)
{
    // Run the row filter for each location in output (which determines
    // the locations to run at) using commandQueue.  input and output are
    // both single-component float images.  filter is a vector of floats.
    // The command will not start until all of waitEvents have completed, and
    // once done will flag doneEvent.


    // Use the pre-allocated output, if given; otherwise, create a new
    // output of appropriate size
    cl::Image2D output
        = (targetImage != nullptr)?
            *targetImage : dummyRun(input);

    const int width = output.getImageInfo<CL_IMAGE_WIDTH>(),
              height = output.getImageInfo<CL_IMAGE_HEIGHT>();


    // Need to work out the filter length; if this value is passed directly,
    // the setArg function doesn't understand its type properly.
    const int filterLength = filter.getInfo<CL_MEM_SIZE>() / sizeof(float);

    // Set all the arguments
    kernel.setArg(0, input);
    kernel.setArg(1, sampler);
    kernel.setArg(2, filter);
    kernel.setArg(3, filterLength);
    kernel.setArg(4, output);

    // Execute
    commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange(width, height),
                                      cl::NullRange,
                                      &waitEvents, doneEvent);

    return output;

}



ColDecimateFilter::ColDecimateFilter(cl::Context& contextArg,
                     const std::vector<cl::Device>& devices)
    : context(contextArg)
{
    // The OpenCL kernel:
    const std::string sourceCode = 
        "__kernel void colDecimateFilter(__read_only image2d_t input,        \n"
        "                                sampler_t inputSampler,             \n"
        "                                __constant float* filter,       \n"
        "                                const int filterLength,             \n"
        "                                __write_only image2d_t output,      \n"
        "                                int offset)                         \n"
        "{                                                                   \n"
        "    // Coordinates in output frame (rows really are rows, and       \n"
        "    // the column is the number of _pairs_ of numbers along (since  \n"  
        "    // the tree outputs are interleaved).                           \n"
        "    int x = get_global_id(0);                                       \n"
        "    int y = get_global_id(1);                                       \n"
        "                                                                    \n"
        "    // Results for each of the two trees                            \n"
        "    float out1 = 0.0f;                                              \n"
        "    float out2 = 0.0f;                                              \n"
        "                                                                    \n"
        "    // Apply the filter forward (for the first tree) and backwards  \n"
        "    // (for the second).                                            \n"
        "    int startY = 4 * y - (filterLength-2) + offset;                 \n"
        "    for (int i = 0; i < filterLength; ++i) {                        \n"
        "        out1 += filter[filterLength-1-i] *                          \n"
        "                read_imagef(input, inputSampler,                    \n"
        "                            (int2) (x, startY+2*i)).x;              \n"
        "        out2 += filter[i] *                                         \n"
        "                read_imagef(input, inputSampler,                    \n"
        "                            (int2) (x, startY+2*i+1)).x;            \n"
        "    }                                                               \n"
        "                                                                    \n"
        "    // Output position is 2*r rows down, plus c along (because      \n"
        "    // the outputs from two trees are interleaved)                  \n"
        "    write_imagef(output, (int2) (x, 2*y), out1);                    \n"
        "    write_imagef(output, (int2) (x, 2*y+1), out2);                  \n"
        "                                                                    \n"
        "                                                                    \n"
        "}                                                                   \n"
        "\n";

    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(std::make_pair(sourceCode.c_str(), sourceCode.length()));

    // Compile it...
    cl::Program program(context, source);
    program.build(devices);
        
    // ...and extract the useful part, viz the kernel
    kernel = cl::Kernel(program, "colDecimateFilter");

    // The sampler, later to be used as a kernel argument
    sampler = createSampler(context);
}


cl::Image2D ColDecimateFilter::dummyRun(const cl::Image2D& input)
{
    return dummyRun(input.getImageInfo<CL_IMAGE_WIDTH>(),
                    input.getImageInfo<CL_IMAGE_HEIGHT>());
}


cl::Image2D ColDecimateFilter::dummyRun(size_t inWidth, size_t inHeight)
{
    bool pad = (inHeight % 4) != 0;

    return createImage2D(context, inWidth, inHeight / 2 + (pad? 1 : 0));
}


cl::Image2D ColDecimateFilter::operator() (cl::CommandQueue& commandQueue,
               cl::Image2D& input, cl::Buffer& filter,
               const std::vector<cl::Event>& waitEvents,
               cl::Event* doneEvent,
               cl::Image2D* targetImage)
{
    // Run the column decimation filter for each location in output (which
    // determines the locations to run at) using commandQueue.  input and 
    // output are both single-component float images.  filter is a vector of
    // floats. The command will not start until all of waitEvents have 
    // completed, and once done will flag doneEvent.


    // Need to work out the filter length; if this value is passed directly,
    // the setArg function doesn't understand its type properly.
    const int filterLength = filter.getInfo<CL_MEM_SIZE>() / sizeof(float);

    // Make sure the resulting image is an even height, i.e. it has the
    // same length for both the trees
    bool pad = (input.getImageInfo<CL_IMAGE_HEIGHT>() % 4) != 0;

    // Use the pre-allocated output, if given; otherwise, create a new
    // output of appropriate size
    cl::Image2D output
        = (targetImage != nullptr)?
            *targetImage : dummyRun(input);

    const int width = output.getImageInfo<CL_IMAGE_WIDTH>(),
              height = output.getImageInfo<CL_IMAGE_HEIGHT>();

    // Tell the kernel to use the buffers, and how long they are
    kernel.setArg(0, input);         // input
    kernel.setArg(1, sampler);       
    kernel.setArg(2, filter);        // filter
    kernel.setArg(3, filterLength);  // filterLength
    kernel.setArg(4, output);        // output
    kernel.setArg(5, pad? -1 : 0);   // Offset to start reading input from

    // Execute
    commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange(width, height / 2),
                                      cl::NullRange,
                                      &waitEvents, doneEvent);

    return output;

}


RowDecimateFilter::RowDecimateFilter(cl::Context& context_,
                     const std::vector<cl::Device>& devices)
   : context(context_)
{
    // The OpenCL kernel:
    const std::string sourceCode = 
        "__kernel void rowDecimateFilter(__read_only image2d_t input,       \n"
        "                                sampler_t inputSampler,            \n"
        "                                __global const float* filter,      \n"
        "                                const int filterLength,            \n"
        "                                __write_only image2d_t output,     \n"
        "                                int offset)                        \n"
        "{                                                                  \n"
        "    // Coordinates in output frame (rows really are rows, and the  \n"
        "    // column is the number of _pairs_ of numbers along (since the \n"
        "    // tree outputs are interleaved).                              \n"
        "    int x = get_global_id(0);                                      \n"
        "    int y = get_global_id(1);                                      \n"
        "                                                                   \n"
        "    // Results for each of the two trees                           \n"
        "    float out1 = 0.0f;                                             \n"
        "    float out2 = 0.0f;                                             \n"
        "                                                                   \n"
        "    // Apply the filter forward (for the first tree) and backwards \n"
        "    // (for the second).                                           \n"
        "    int startX = 4 * x - (filterLength-2) + offset;                \n"
        "    for (int i = 0; i < filterLength; ++i) {                       \n"
        "        out1 += filter[filterLength-1-i] *                         \n"
        "                read_imagef(input, inputSampler,                   \n"
        "                            (int2) (startX+2*i, y)).x;             \n"
        "        out2 += filter[i] *                                        \n"
        "                read_imagef(input, inputSampler,                   \n"
        "                            (int2) (startX+2*i+1, y)).x;           \n"
        "    }                                                              \n"
        "                                                                   \n"
        "    // Output position is r rows down, plus 2*c along (because the \n"
        "    // outputs from two trees are interleaved)                     \n"
        "    write_imagef(output, (int2) (2*x,   y), out1);                 \n"
        "    write_imagef(output, (int2) (2*x+1, y), out2);                 \n"
        "}                                                                  \n"
        "\n";

    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(std::make_pair(sourceCode.c_str(), sourceCode.length()));

    // Compile it...
    cl::Program program(context, source);
    program.build(devices);
        
    // ...and extract the useful part, viz the kernel
    kernel = cl::Kernel(program, "rowDecimateFilter");

    // The sampler, later to be used as a kernel argument
    sampler = createSampler(context);
}


cl::Image2D RowDecimateFilter::dummyRun(const cl::Image2D& input)
{
    return dummyRun(input.getImageInfo<CL_IMAGE_WIDTH>(),
                    input.getImageInfo<CL_IMAGE_HEIGHT>());
}

cl::Image2D RowDecimateFilter::dummyRun(size_t inWidth, size_t inHeight)
{
    bool pad = (inWidth % 4) != 0;

    return createImage2D(context, inWidth / 2 + (pad? 1 : 0), inHeight);
}




cl::Image2D RowDecimateFilter::operator() (cl::CommandQueue& commandQueue,
               cl::Image2D& input, 
               cl::Buffer& filter,
               const std::vector<cl::Event>& waitEvents,
               cl::Event* doneEvent,
               cl::Image2D* targetImage)
{
    // Run the row decimation filter for each location in output (which
    // determines the locations to run at) using commandQueue.  input and 
    // output are both single-component float images.  filter is a vector of
    // floats. The command will not start until all of waitEvents have 
    // completed, and once done will flag doneEvent.

    // Make sure the resulting image is an even height, i.e. it has the
    // same length for both the trees
    bool pad = (input.getImageInfo<CL_IMAGE_WIDTH>() % 4) != 0;

    cl::Image2D output
        = (targetImage != nullptr) ?
            *targetImage : dummyRun(input);

    const int width = output.getImageInfo<CL_IMAGE_WIDTH>(),
              height = output.getImageInfo<CL_IMAGE_HEIGHT>();

    // Need to work out the filter length; if this value is passed directly,
    // the setArg function doesn't understand its type properly.
    const int filterLength = filter.getInfo<CL_MEM_SIZE>() / sizeof(float);

    // Tell the kernel to use the buffers, and how long they are
    kernel.setArg(0, input);         // input
    kernel.setArg(1, sampler);       
    kernel.setArg(2, filter);        // filter
    kernel.setArg(3, filterLength);  // filterLength
    kernel.setArg(4, output);        // output
    kernel.setArg(5, pad? -1 : 0);

    // Execute
    commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange(width / 2, height),
                                      cl::NullRange,
                                      &waitEvents, doneEvent);

    return output;
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
"__kernel void quadToComplex(__read_only image2d_t input,                  \n"
"                          sampler_t inputSampler,                         \n"
"                          __write_only image2d_t out1,                    \n"
"                          __write_only image2d_t out2)                    \n"
"{                                                                         \n"
"    int x = get_global_id(0);                                             \n"
"    int y = get_global_id(1);                                             \n"
"                                                                          \n"
"    // Sample upper left, upper right, etc                                \n"
"    float ul = read_imagef(input, inputSampler, (int2) (  2*x,   2*y)).x; \n"
"    float ur = read_imagef(input, inputSampler, (int2) (2*x+1,   2*y)).x; \n"
"    float ll = read_imagef(input, inputSampler, (int2) (  2*x, 2*y+1)).x; \n"
"    float lr = read_imagef(input, inputSampler, (int2) (2*x+1, 2*y+1)).x; \n"
"                                                                          \n"
"    const float factor = 1.0f / sqrt(2.0f);                               \n"
"                                                                          \n"
"    // Combine into complex pairs                                         \n"
"    write_imagef(out1, (int2) (x, y),                                     \n"
"                       factor * (float4) (ul - lr, ur + ll, 0, 0));             \n"
"    write_imagef(out2, (int2) (x, y),                                     \n"
"                       factor * (float4) (ul + lr, ur - ll, 0, 0));             \n"
"}                                                                         \n"
"\n";

    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(std::make_pair(sourceCode.c_str(), sourceCode.length()));

    // Compile it...
    cl::Program program(context, source);
    program.build(devices);
        
    // ...and extract the useful part, viz the kernel
    kernel = cl::Kernel(program, "quadToComplex");

    // The sampler, later to be used as a kernel argument
    sampler = createSampler(context);
}


cl::Image2D QuadToComplex::dummyRun(const cl::Image2D& input)
{
    return dummyRun(input.getImageInfo<CL_IMAGE_WIDTH>(),
                    input.getImageInfo<CL_IMAGE_HEIGHT>());
}

cl::Image2D QuadToComplex::dummyRun(size_t inWidth, size_t inHeight)
{
    return {context, 0, {CL_RG, CL_FLOAT}, inWidth / 2, inHeight / 2};
}




std::tuple<cl::Image2D, cl::Image2D>
QuadToComplex::operator() (cl::CommandQueue& commandQueue,
               cl::Image2D& input,
               const std::vector<cl::Event>& waitEvents,
               cl::Event* doneEvent,
               cl::Image2D* target1, cl::Image2D* target2)
{
    // Outputs are images with two floats per location: (real, imag)
    // Allocate new output images if they weren't passed in
    cl::Image2D out1 = target1? *target1 : dummyRun(input);
    cl::Image2D out2 = target2? *target2 : dummyRun(input);

    // Set up all the arguments to the kernel
    kernel.setArg(0, input);
    kernel.setArg(1, sampler);
    kernel.setArg(2, out1);
    kernel.setArg(3, out2);

    // Create outputs of the correct size
    const int width  = out1.getImageInfo<CL_IMAGE_WIDTH>();
    const int height = out1.getImageInfo<CL_IMAGE_HEIGHT>();

    // Execute
    commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange(width, height),
                                      cl::NullRange,
                                      &waitEvents, doneEvent);

    return std::make_tuple(out1, out2);
}





