#include "filterer.h"
#include <stdexcept>
#include <iostream>
#include <string>
#include <fstream>


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
    sampler = cl::Sampler(context, false, CL_ADDRESS_CLAMP_TO_EDGE,
                          CL_FILTER_NEAREST);
}


void RowFilter::operator() (cl::CommandQueue& commandQueue,
               cl::Image2D& output, cl::Image2D& input, 
               cl::Buffer& filter)
{
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
                                      cl::NullRange);

}

#include <iostream>

void rowFilter(cl::Context& context,
               cl::CommandQueue& commandQueue,
               cl::Kernel& rowFilterKernel,
               cl::Image2D& output, cl::Image2D& input, 
               cl::Buffer& filter)
{
    int filterLength = filter.getInfo<CL_MEM_SIZE>() / sizeof(float);

    // Tell the kernel to use the buffers, and how long they are
    rowFilterKernel.setArg(0, input);         // input
    rowFilterKernel.setArg(1, createSampler(context));       // inputStride
    rowFilterKernel.setArg(2, filter);     // filter
    rowFilterKernel.setArg(3, filterLength);     // filterLength
    rowFilterKernel.setArg(4, output);        // output

    std::cout << input.getImageInfo<CL_IMAGE_WIDTH>()
              << " "
              << input.getImageInfo<CL_IMAGE_HEIGHT>()
              << std::endl;

    std::cout << output.getImageInfo<CL_IMAGE_WIDTH>()
              << " "
              << output.getImageInfo<CL_IMAGE_HEIGHT>()
              << std::endl;

    // Output size
    const int rows = output.getImageInfo<CL_IMAGE_HEIGHT>();
    const int cols = output.getImageInfo<CL_IMAGE_WIDTH>();

    // Execute
    commandQueue.enqueueNDRangeKernel(rowFilterKernel, cl::NullRange,
                                      cl::NDRange(360, 500),
                                      cl::NullRange);

    commandQueue.finish();
}





cl::Image2D createImage2D(cl::Context& context, 
                          cl::CommandQueue& commandQueue,
                          int width, int height)
{
    cl::Image2D outImage(context,
                       0,
                       cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), 
                       width, height);
    //commandQueue.finish();
    return outImage;
}





cl::Buffer createBuffer(cl::Context& context,
                        cl::CommandQueue& commandQueue, 
                        const float data[], int length)
{
    //CL_MEM_COPY_HOST_PTR
    cl::Buffer buffer(context, CL_MEM_READ_WRITE,
                        sizeof(float) * length
                        );

    commandQueue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(float) * length,
                           const_cast<float*>(data));
    commandQueue.finish();

    return buffer;
}



cl::Sampler createSampler(cl::Context& context)
{
    return cl::Sampler(context, CL_FALSE, CL_ADDRESS_CLAMP,
                       CL_FILTER_NEAREST);
}



void rowDecimateFilter(cl::Context& context,
                       cl::CommandQueue& commandQueue,
                       cl::Kernel& rowDecimateFilterKernel,
                       cl::Image2D& output, cl::Image2D& input, 
                       cl::Buffer& filter, bool pad)
{
    int filterLength = filter.getInfo<CL_MEM_SIZE>() / sizeof(float);

    // Tell the kernel to use the buffers, and how long they are
    rowDecimateFilterKernel.setArg(0, input);         // input
    rowDecimateFilterKernel.setArg(1, createSampler(context));    
    // inputStride
    rowDecimateFilterKernel.setArg(2, filter);     // filter
    rowDecimateFilterKernel.setArg(3, filterLength);     // filterLength
    rowDecimateFilterKernel.setArg(4, output);        // output
    rowDecimateFilterKernel.setArg(5, pad? -1 : 0);

    // Output size
    const int rows = output.getImageInfo<CL_IMAGE_HEIGHT>();
    const int cols = output.getImageInfo<CL_IMAGE_WIDTH>() / 2;

    // Execute
    commandQueue.enqueueNDRangeKernel(rowDecimateFilterKernel, cl::NullRange,
                                      cl::NDRange(cols, rows),
                                      cl::NullRange);

    commandQueue.finish();
}


void colDecimateFilter(cl::Context& context,
                       cl::CommandQueue& commandQueue,
                       cl::Kernel& colDecimateFilterKernel,
                       cl::Image2D& output, cl::Image2D& input, 
                       cl::Buffer& filter, bool pad)
{
    int filterLength = filter.getInfo<CL_MEM_SIZE>() / sizeof(float);

    // Tell the kernel to use the buffers, and how long they are
    colDecimateFilterKernel.setArg(0, input);         // input
    colDecimateFilterKernel.setArg(1, createSampler(context));       // inputStride
    colDecimateFilterKernel.setArg(2, filter);     // filter
    colDecimateFilterKernel.setArg(3, filterLength);     // filterLength
    colDecimateFilterKernel.setArg(4, output);        // output
    colDecimateFilterKernel.setArg(5, pad? -1 : 0);

    // Output size
    const int rows = output.getImageInfo<CL_IMAGE_HEIGHT>() / 2;
    const int cols = output.getImageInfo<CL_IMAGE_WIDTH>();

    // Execute
    commandQueue.enqueueNDRangeKernel(colDecimateFilterKernel, cl::NullRange,
                                      cl::NDRange(cols, rows),
                                      cl::NullRange);
    commandQueue.finish();
}


void colFilter(cl::Context& context,
               cl::CommandQueue& commandQueue,
               cl::Kernel& colFilterKernel,
               cl::Image2D& output, cl::Image2D& input, 
               cl::Buffer& filter)
{
    int filterLength = filter.getInfo<CL_MEM_SIZE>() / sizeof(float);

    // Tell the kernel to use the buffers, and how long they are
    colFilterKernel.setArg(0, input);         // input
    colFilterKernel.setArg(1, createSampler(context));       // inputStride
    colFilterKernel.setArg(2, filter);     // filter
    colFilterKernel.setArg(3, filterLength);     // filterLength
    colFilterKernel.setArg(4, output);        // output

    // Output size
    const int rows = output.getImageInfo<CL_IMAGE_HEIGHT>();
    const int cols = output.getImageInfo<CL_IMAGE_WIDTH>();

    // Execute
    commandQueue.enqueueNDRangeKernel(colFilterKernel, cl::NullRange,
                                      cl::NDRange(cols, rows),
                                      cl::NullRange);

    commandQueue.finish();
}


void quadToComplex(cl::Context& context,
                   cl::CommandQueue& commandQueue,
                   cl::Kernel& quadToComplexKernel,
                   cl::Image2D& out1Re, cl::Image2D& out1Im,
                   cl::Image2D& out2Re, cl::Image2D& out2Im,
                   cl::Image2D& input)
{
    quadToComplexKernel.setArg(0, input);
    quadToComplexKernel.setArg(1, createSampler(context));
    quadToComplexKernel.setArg(2, out1Re);
    quadToComplexKernel.setArg(3, out1Im);
    quadToComplexKernel.setArg(4, out2Re);
    quadToComplexKernel.setArg(5, out2Im);

    // Output size
    const int rows = out1Re.getImageInfo<CL_IMAGE_HEIGHT>();
    const int cols = out1Re.getImageInfo<CL_IMAGE_WIDTH>();

    // Execute
    commandQueue.enqueueNDRangeKernel(quadToComplexKernel, cl::NullRange,
                                      cl::NDRange(cols, rows),
                                      cl::NullRange);
    commandQueue.finish();
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
    output = createImage2D(context, commandQueue, width, height);

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

