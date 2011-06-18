#include <iostream>
#include <fstream>
#include <vector>
#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"
#include "highgui.h"
#include <stdexcept>

class Filterer {
public:
    Filterer();

    cv::Mat colDecimateFilter(const cv::Mat& input, 
                              const float* filter, int filterLength);
    cv::Mat rowDecimateFilter(const cv::Mat& input,
                              const float* filter, int filterLength);

private:

    cl::Context context;
    cl::Program program;
    cl::Kernel kernel;
    cl::CommandQueue commandQueue;

    cl::Kernel rowDecimateFilterKernel;
    cl::Kernel colDecimateFilterKernel;
};

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
    program.build(devices);

    // Turn these into kernels
    kernel = cl::Kernel(program, "square");
    rowDecimateFilterKernel = cl::Kernel(program, "rowDecimateFilter");
    colDecimateFilterKernel = cl::Kernel(program, "colDecimateFilter");

    // Ready the command queue on the first device to hand
    commandQueue = cl::CommandQueue(context, devices[0]);
}


cv::Mat Filterer::colDecimateFilter(const cv::Mat& input,
                                    const float* filter, int filterLength)
{
    // Check we can deal with the input
    if (!input.isContinuous())
        throw std::runtime_error("Input to Filterer is not contiguous");

    cv::Mat output((input.rows+1 - filterLength) / 2, input.cols, 
                   input.type());

    // Create buffers
    cl::Buffer inBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        input.elemSize() * input.total(),
                        const_cast<uchar*>(input.ptr()));
    // Yes, the const cast is unpleasant, but the buffer is only ever going
    // to be read (hence the exception).
    
    cl::Buffer outBuffer(context, CL_MEM_WRITE_ONLY,
                         output.elemSize() * output.total()); 


    cl::Buffer filterBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * filterLength,
                        const_cast<float*>(filter));

    // Tell the kernel to use the buffers, and how long they are
    colDecimateFilterKernel.setArg(0, inBuffer);         // input
    colDecimateFilterKernel.setArg(1, input.cols);       // inputStride
    colDecimateFilterKernel.setArg(2, filterBuffer);     // filter
    colDecimateFilterKernel.setArg(3, filterLength);     // filterLength
    colDecimateFilterKernel.setArg(4, outBuffer);        // output
    colDecimateFilterKernel.setArg(5, output.cols);      // outputStride


    // Execute
    commandQueue.enqueueNDRangeKernel(colDecimateFilterKernel, cl::NullRange,
                                      cl::NDRange(output.rows/2,
                                                  output.cols),
                                      cl::NullRange);

    // Extract results
    commandQueue.enqueueReadBuffer(outBuffer, CL_TRUE,
                                   0, output.elemSize() * output.total(),
                                   output.ptr());

    return output;
}


cv::Mat Filterer::rowDecimateFilter(const cv::Mat& input, 
                                    const float* filter, int filterLength)
{
    // Check we can deal with the input
    if (!input.isContinuous())
        throw std::runtime_error("Input to Filterer is not contiguous");

    cv::Mat output(input.rows, (input.cols+1 - filterLength) / 2, 
                   input.type());

    // Create buffers
    cl::Buffer inBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        input.elemSize() * input.total(),
                        const_cast<uchar*>(input.ptr()));
    // Yes, the const cast is unpleasant, but the buffer is only ever going
    // to be read (hence the exception).
    
    cl::Buffer outBuffer(context, CL_MEM_WRITE_ONLY,
                         output.elemSize() * output.total()); 


    cl::Buffer filterBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * filterLength,
                        const_cast<float*>(filter));

    // Tell the kernel to use the buffers, and how long they are
    rowDecimateFilterKernel.setArg(0, inBuffer);         // input
    rowDecimateFilterKernel.setArg(1, input.cols);       // inputStride
    rowDecimateFilterKernel.setArg(2, filterBuffer);     // filter
    rowDecimateFilterKernel.setArg(3, filterLength);     // filterLength
    rowDecimateFilterKernel.setArg(4, outBuffer);        // output
    rowDecimateFilterKernel.setArg(5, output.cols);      // outputStride


    // Execute
    commandQueue.enqueueNDRangeKernel(rowDecimateFilterKernel, cl::NullRange,
                                      cl::NDRange(output.rows,
                                                  output.cols/2),
                                      cl::NullRange);

    // Extract results
    commandQueue.enqueueReadBuffer(outBuffer, CL_TRUE,
                                   0, output.elemSize() * output.total(),
                                   output.ptr());

    return output;
}






int main()
{
    cv::Mat inImage = cv::imread("test.bmp", 0);
    cv::Mat input;
    inImage.convertTo(input, CV_32F);


    try {
        Filterer filterer;

        // Send over the filter
        const int filterLength = 8;
        const float filter[filterLength] = {0.0f, 0.0f, 0.0f, 1.0f,
                                       -0.0f, -0.0f,  -0.0f, -0.0f};
        cv::Mat filteredImage = filterer.rowDecimateFilter(
            filterer.colDecimateFilter(input, filter, filterLength),
            filter, filterLength);
    
        cv::namedWindow("Output", CV_WINDOW_AUTOSIZE);
        cv::imshow("Output", filteredImage);
    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }

    cv::waitKey();
                     
    return 0;
}

