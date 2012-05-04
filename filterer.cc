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
    return cl::Sampler(context, CL_FALSE, CL_ADDRESS_MIRRORED_REPEAT,
                       CL_FILTER_NEAREST);
}





Filter::Filter(cl::Context& context,
               const std::vector<cl::Device>& devices,
               cl::Buffer coefficients,
               int dimension)
   : context_(context), coefficients_(coefficients), dimension_(dimension),
     wgSize0_(1), wgSize1_(16)
{
    // The OpenCL kernel:
    std::ostringstream kernelInput;

    // Need to work out the filter length; if this value is passed directly,
    // the setArg function doesn't understand its type properly.
    const int filterLength = coefficients_.getInfo<CL_MEM_SIZE>() 
                                / sizeof(float);

    const int inputLocalSize =        
        (dimension == 0) ?
            ((wgSize0_ + filterLength - 1) * wgSize1_)
           :((wgSize1_ + filterLength - 1) * wgSize0_);

    kernelInput
    << "__kernel void filter(__read_only image2d_t input,           \n"
        "                        __write_only image2d_t output,         \n"
        "                        __constant float* filter)          \n"
        "{                                                              \n"
        "sampler_t inputSampler ="
            "CLK_NORMALIZED_COORDS_FALSE"
            "| CLK_ADDRESS_MIRRORED_REPEAT"
            "| CLK_FILTER_NEAREST;"

        "__local float inputLocal[" << inputLocalSize << "];"

        "const int filterLength = " << filterLength << ";"

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
    kernel_ = cl::Kernel(program, "filter");

    // We know what the filter argument will be already
    kernel_.setArg(2, coefficients_);
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

    const int width = output.getImageInfo<CL_IMAGE_WIDTH>(),
              height = output.getImageInfo<CL_IMAGE_HEIGHT>();

    // Need to work out the filter length; if this value is passed directly,
    // the setArg function doesn't understand its type properly.
    const int filterLength = coefficients_.getInfo<CL_MEM_SIZE>() 
                                / sizeof(float);

    int heightWG = 64;
    cl::NDRange WorkgroupSize = {1, heightWG};
    int numVertWGs = height / heightWG + ((height % heightWG) ? 1 : 0); 
    cl::NDRange GlobalSize = {width, heightWG * numVertWGs }; 

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
    if (dimension_ == 0)
        return createImage2D(context_, inWidth + inWidth % 2, inHeight);
    else
        return createImage2D(context_, inWidth, inHeight + inHeight % 2);
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





