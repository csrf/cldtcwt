#include "filterer.h"
#include <stdexcept>
#include <iostream>
#include <fstream>

Filterer::Filterer()
{
    // Retrive platform information
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.size() == 0)
        throw std::runtime_error("No platforms!");

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    // Create a context to work in 
    context = cl::Context(devices);


    // Open the program, find its length and read it out
    std::ifstream sourceFile("kernel.cl", std::ios::in | std::ios::ate);
    std::string kernelSource(sourceFile.tellg(), ' ');
    sourceFile.seekg(0, std::ios::beg);
    sourceFile.read(&kernelSource[0], kernelSource.length());

    // Create a program compiled from the source code (read in previously)
    cl::Program::Sources source;
    source.push_back(std::pair<const char*, size_t>(kernelSource.c_str(),
                                kernelSource.length()));
    program = cl::Program(context, source);
    try {
        program.build(devices);
    } catch(cl::Error err) {
        std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) 
                  << std::endl;
        throw;
    }

    // Turn these into kernels
    rowDecimateFilterKernel = cl::Kernel(program, "rowDecimateFilter");
    colDecimateFilterKernel = cl::Kernel(program, "colDecimateFilter");
    rowFilterKernel = cl::Kernel(program, "rowFilter");
    colFilterKernel = cl::Kernel(program, "colFilter");
    quadToComplexKernel = cl::Kernel(program, "quadToComplex");

    // Ready the command queue on the first device to hand
    commandQueue = cl::CommandQueue(context, devices[0]);
}
    



cl::Image2D Filterer::createImage2D(cv::Mat& image)
{
    cl::Image2D outImage(context,
                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        cl::ImageFormat(CL_RGBA, CL_FLOAT), 
                        image.cols, image.rows, 0,
                        image.ptr());

    commandQueue.finish();

    return outImage;
}


cl::Image2D Filterer::createImage2D(int width, int height)
{
    cl::Image2D outImage(context,
                       CL_MEM_READ_WRITE,
                       cl::ImageFormat(CL_RGBA, CL_FLOAT), 
                       width, height);
    commandQueue.finish();
    return outImage;
}



cv::Mat Filterer::getImage2D(cl::Image2D image)
{
    // Create a matrix to put the data into
    cv::Mat output(image.getImageInfo<CL_IMAGE_HEIGHT>(), 
                   image.getImageInfo<CL_IMAGE_WIDTH>(),
                   CV_32FC4);


    // Read the data out, blocking until done.  Possible scope for
    // optimisation at a later date.
    //
    cl::size_t<3> origin, extents;
    origin.push_back(0);
    origin.push_back(0);
    origin.push_back(0);
    extents.push_back(output.cols);
    extents.push_back(output.rows);
    extents.push_back(1);


    commandQueue.enqueueReadImage(image, CL_TRUE,
                                  origin, extents,                                                                0, 0,
                                  output.ptr());

    return output;
}

cl::Buffer Filterer::createBuffer(const float data[], int length)
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

cl::Sampler Filterer::createSampler()
{
    return cl::Sampler(context, CL_FALSE, CL_ADDRESS_CLAMP,
                       CL_FILTER_NEAREST);
}


void Filterer::rowDecimateFilter(cl::Image2D& output, cl::Image2D& input, 
                                    cl::Buffer& filter, bool pad)
{
    int filterLength = filter.getInfo<CL_MEM_SIZE>() / sizeof(float);

    // Tell the kernel to use the buffers, and how long they are
    rowDecimateFilterKernel.setArg(0, input);         // input
    rowDecimateFilterKernel.setArg(1, createSampler());       // inputStride
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


void Filterer::colDecimateFilter(cl::Image2D& output, cl::Image2D& input, 
                                  cl::Buffer& filter, bool pad)
{
    int filterLength = filter.getInfo<CL_MEM_SIZE>() / sizeof(float);

    // Tell the kernel to use the buffers, and how long they are
    colDecimateFilterKernel.setArg(0, input);         // input
    colDecimateFilterKernel.setArg(1, createSampler());       // inputStride
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


void Filterer::rowFilter(cl::Image2D& output, cl::Image2D& input, 
                                    cl::Buffer& filter)
{
    int filterLength = filter.getInfo<CL_MEM_SIZE>() / sizeof(float);

    // Tell the kernel to use the buffers, and how long they are
    rowFilterKernel.setArg(0, input);         // input
    rowFilterKernel.setArg(1, createSampler());       // inputStride
    rowFilterKernel.setArg(2, filter);     // filter
    rowFilterKernel.setArg(3, filterLength);     // filterLength
    rowFilterKernel.setArg(4, output);        // output

    // Output size
    const int rows = output.getImageInfo<CL_IMAGE_HEIGHT>();
    const int cols = output.getImageInfo<CL_IMAGE_WIDTH>();

    // Execute
    commandQueue.enqueueNDRangeKernel(rowFilterKernel, cl::NullRange,
                                      cl::NDRange(cols, rows),
                                      cl::NullRange);

    commandQueue.finish();
}


void Filterer::colFilter(cl::Image2D& output, cl::Image2D& input, 
                                    cl::Buffer& filter)
{
    int filterLength = filter.getInfo<CL_MEM_SIZE>() / sizeof(float);

    // Tell the kernel to use the buffers, and how long they are
    colFilterKernel.setArg(0, input);         // input
    colFilterKernel.setArg(1, createSampler());       // inputStride
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


void Filterer::quadToComplex(cl::Image2D& out1Re, cl::Image2D& out1Im,
                             cl::Image2D& out2Re, cl::Image2D& out2Im,
                             cl::Image2D& input)
{
    quadToComplexKernel.setArg(0, input);
    quadToComplexKernel.setArg(1, createSampler());
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


