#include "filterer.h"
#include "clUtil.h"
#include <stdexcept>
#include <iostream>
#include <string>
#include <fstream>

// Central sampler creating function (to make changing the addressing
// overflow behaviour easy, for upgrade to OpenCL 1.1)
cl::Sampler createSampler(cl::Context& context)
{
    return cl::Sampler(context, CL_FALSE, CL_ADDRESS_CLAMP,
                       CL_FILTER_NEAREST);
}


ColFilter::ColFilter(cl::Context& context_,
                     const std::vector<cl::Device>& devices)
   : context(context_)
{
    // The OpenCL kernel:
    const std::string sourceCode = 
        "__kernel void colFilter(__read_only image2d_t input,           \n"
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
        "    int startY = y - (filterLength-1) / 2;                     \n"
        "    for (int i = 0; i < filterLength; ++i)                     \n"
        "        out += filter[filterLength-1-i] *                      \n"
        "                read_imagef(input, inputSampler,               \n"
        "                            (int2) (x, startY + i)).x;         \n"
        "                                                               \n"
        "    // Output position is r rows down, plus 2*c along (because \n"
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
    kernel = cl::Kernel(program, "colFilter");

    // The sampler, later to be used as a kernel argument
    sampler = createSampler(context);
}



void ColFilter::operator() (cl::CommandQueue& commandQueue,
               cl::Image2D& output, cl::Image2D& input, 
               cl::Buffer& filter,
               const std::vector<cl::Event>* waitEvents,
               cl::Event* doneEvent)
{
    // Run the column filter for each location in output (which determines
    // the locations to run at) using commandQueue.  input and output are
    // both single-component float images.  filter is a vector of floats.
    // The command will not start until all of waitEvents have completed, and
    // once done will flag doneEvent.


    // Need to work out the filter length; if this value is passed directly,
    // the setArg function doesn't understand its type properly.
    const int filterLength = filter.getInfo<CL_MEM_SIZE>() / sizeof(float);

    // Set all the arguments
    kernel.setArg(0, input);
    kernel.setArg(1, sampler);
    kernel.setArg(2, filter);
    kernel.setArg(3, filterLength);
    kernel.setArg(4, output);

    // Output size
    const int width = output.getImageInfo<CL_IMAGE_WIDTH>();
    const int height = output.getImageInfo<CL_IMAGE_HEIGHT>();

    // Execute
    commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange(width, height),
                                      cl::NullRange,
                                      waitEvents, doneEvent);

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


void RowFilter::operator() (cl::CommandQueue& commandQueue,
               cl::Image2D& output, cl::Image2D& input, 
               cl::Buffer& filter,
               const std::vector<cl::Event>* waitEvents,
               cl::Event* doneEvent)
{
    // Run the row filter for each location in output (which determines
    // the locations to run at) using commandQueue.  input and output are
    // both single-component float images.  filter is a vector of floats.
    // The command will not start until all of waitEvents have completed, and
    // once done will flag doneEvent.


    // Need to work out the filter length; if this value is passed directly,
    // the setArg function doesn't understand its type properly.
    const int filterLength = filter.getInfo<CL_MEM_SIZE>() / sizeof(float);

    // Set all the arguments
    kernel.setArg(0, input);
    kernel.setArg(1, sampler);
    kernel.setArg(2, filter);
    kernel.setArg(3, filterLength);
    kernel.setArg(4, output);

    // Output size
    const int width = output.getImageInfo<CL_IMAGE_WIDTH>();
    const int height = output.getImageInfo<CL_IMAGE_HEIGHT>();

    // Execute
    commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange(width, height),
                                      cl::NullRange,
                                      waitEvents, doneEvent);

}



ColDecimateFilter::ColDecimateFilter(cl::Context& contextArg,
                     const std::vector<cl::Device>& devices)
    : context(contextArg)
{
    // The OpenCL kernel:
    const std::string sourceCode = 
        "__kernel void colDecimateFilter(__read_only image2d_t input,        \n"
        "                                sampler_t inputSampler,             \n"
        "                                __global const float* filter,       \n"
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



cl::Image2D ColDecimateFilter::operator() (cl::CommandQueue& commandQueue,
               cl::Image2D& input, cl::Buffer& filter,
               const std::vector<cl::Event>& waitEvents,
               cl::Event* doneEvent)
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
    int inHeight = input.getImageInfo<CL_IMAGE_HEIGHT>();
    bool pad = (inHeight % 4) != 0;

    // Create the output at the right size
    const int width = input.getImageInfo<CL_IMAGE_WIDTH>(),
              height = inHeight / 2 + (pad? 1 : 0);
    cl::Image2D output = createImage2D(context, width, height);

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



void RowDecimateFilter::operator() (cl::CommandQueue& commandQueue,
               cl::Image2D& output, cl::Image2D& input, 
               cl::Buffer& filter,
               bool pad,
               const std::vector<cl::Event>* waitEvents,
               cl::Event* doneEvent)
{
    // Run the row decimation filter for each location in output (which
    // determines the locations to run at) using commandQueue.  input and 
    // output are both single-component float images.  filter is a vector of
    // floats. The command will not start until all of waitEvents have 
    // completed, and once done will flag doneEvent.


    // Need to work out the filter length; if this value is passed directly,
    // the setArg function doesn't understand its type properly.
    const int filterLength = filter.getInfo<CL_MEM_SIZE>() / sizeof(float);

    // Tell the kernel to use the buffers, and how long they are
    kernel.setArg(0, input);         // input
    kernel.setArg(1, createSampler(context));       
                                     // inputStride
    kernel.setArg(2, filter);        // filter
    kernel.setArg(3, filterLength);  // filterLength
    kernel.setArg(4, output);        // output
    kernel.setArg(5, pad? -1 : 0);

    // Output size
    const int height = output.getImageInfo<CL_IMAGE_HEIGHT>();
    const int width  = output.getImageInfo<CL_IMAGE_WIDTH>();

    // Execute
    commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange(width / 2, height),
                                      cl::NullRange,
                                      waitEvents, doneEvent);

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
"                          __write_only image2d_t out1Re,                  \n"
"                          __write_only image2d_t out1Im,                  \n"
"                          __write_only image2d_t out2Re,                  \n"
"                          __write_only image2d_t out2Im)                  \n"
"{                                                                         \n"
"    int x = get_global_id(0);                                             \n"
"    int y = get_global_id(1);                                             \n"
"                                                                          \n"
"    const float factor = 1.0f / sqrt(2.0f);                               \n"
"                                                                          \n"
"    // Sample upper left, upper right, etc                                \n"
"    float ul = read_imagef(input, inputSampler, (int2) (  2*x,   2*y)).x; \n"
"    float ur = read_imagef(input, inputSampler, (int2) (2*x+1,   2*y)).x; \n"
"    float ll = read_imagef(input, inputSampler, (int2) (  2*x, 2*y+1)).x; \n"
"    float lr = read_imagef(input, inputSampler, (int2) (2*x+1, 2*y+1)).x; \n"
"                                                                          \n"
"    write_imagef(out1Re, (int2) (x,y), factor * (ul - lr));               \n"
"    write_imagef(out1Im, (int2) (x,y), factor * (ur + ll));               \n"
"    write_imagef(out2Re, (int2) (x,y), factor * (ul + lr));               \n"
"    write_imagef(out2Im, (int2) (x,y), factor * (ur - ll));               \n"
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




void QuadToComplex::operator() (cl::CommandQueue& commandQueue,
               cl::Image2D& out1Re, cl::Image2D& out1Im,
               cl::Image2D& out2Re, cl::Image2D& out2Im,
               cl::Image2D& input,
               const std::vector<cl::Event>* waitEvents,
               cl::Event* doneEvent)
{

    // Set up all the arguments to the kernel
    kernel.setArg(0, input);
    kernel.setArg(1, sampler);
    kernel.setArg(2, out1Re);
    kernel.setArg(3, out1Im);
    kernel.setArg(4, out2Re);
    kernel.setArg(5, out2Im);

    // Output size
    const int width  = out1Re.getImageInfo<CL_IMAGE_WIDTH>();
    const int height = out1Re.getImageInfo<CL_IMAGE_HEIGHT>();

    // Execute
    commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange(width, height),
                                      cl::NullRange,
                                      waitEvents, doneEvent);

}





