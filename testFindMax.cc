#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

#include "filterer.h"
#include "clUtil.h"
#include <iomanip>

#include <stdexcept>

#include <highgui.h>


#include "findMax.h"

std::tuple<cl::Platform, std::vector<cl::Device>, 
           cl::Context, cl::CommandQueue> 
    initOpenCL();

cl::Image2D createImage2D(cl::Context& context, cv::Mat& mat);




int main()
{
    try {

        cl::Platform platform;
        std::vector<cl::Device> devices;
        cl::Context context;
        cl::CommandQueue commandQueue; 
        std::tie(platform, devices, context, commandQueue) = initOpenCL();

        //-----------------------------------------------------------------
        // Starting test code
        
        FindMax findMax(context, devices);

        const int width = 40, height = 40;
        // Set up data for the input image
        float data[height][width];
        for (int x = 0; x < width; ++x)
            for (int y = 0; y < height; ++y)
                data[y][x] = 0.0f;
        //data[10][5] = 1.0f;
        data[15][15] = 1.0f;
        data[16][15] = 0.5f;
        data[18][12] = 0.05f;
        data[13][15] = 1.0f;

        data[24][30] = 0.5f;

        cl::Image2D inImage = {
            context, 
            CL_MEM_READ_WRITE,
            cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), 
            width, height, 0
        };


        writeImage2D(commandQueue, inImage, &data[0][0]);
        commandQueue.finish();
                                       

        int zero = 0;

        cl::Buffer outputs = {
            context,
            0,              // Flags
            10 * 2 * sizeof(float) // Size to allocate
        };

        cl::Buffer numOutputs = {
            context,
            CL_MEM_COPY_HOST_PTR,              // Flags
            sizeof(int), // Size to allocate
            &zero
        };

        cl::Buffer lock = {
            context,
            CL_MEM_COPY_HOST_PTR,              // Flags
            sizeof(int), // Size to allocate
            &zero 
        };

        float zerof = 0.5f;

        cl::Image2D zeroImg = {
            context, 
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), 
            1, 1, 0,
            &zerof
        };


        findMax(commandQueue, inImage, zeroImg, zeroImg, 
                0.1f, outputs, numOutputs, lock);
        commandQueue.finish();

        int numOutputsVal;
        commandQueue.enqueueReadBuffer(numOutputs, true, 0, sizeof(int),
                                       &numOutputsVal);

        std::cout << numOutputsVal << " outputs" << std::endl;

        numOutputsVal = (numOutputsVal > 10) ? 10 : numOutputsVal;
        
        if (numOutputsVal > 0) {
            std::vector<float> results(numOutputsVal * 2);
            commandQueue.enqueueReadBuffer(outputs, true, 0, 
                                           numOutputsVal * 2 * sizeof(float),
                                           &results[0]);

            for (auto v: results)
                std::cout << v << std::endl;
        }

    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
                     
    return 0;
}


std::tuple<cl::Platform, std::vector<cl::Device>, 
           cl::Context, cl::CommandQueue> 
initOpenCL()
{
    // Get platform, devices, command queue

    // Retrive platform information
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.size() == 0)
        throw std::runtime_error("No platforms!");

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);

    // Create a context to work in 
    cl::Context context(devices);

    // Ready the command queue on the first device to hand
    cl::CommandQueue commandQueue(context, devices[0]);

    return std::make_tuple(platforms[0], devices, context, commandQueue);
}


cl::Image2D createImage2D(cl::Context& context, cv::Mat& mat)
{
    if (mat.type() == CV_32F) {
        // If in the right format already, just create the image and point
        // it to the data
        return cl::Image2D(context, 
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), 
                           mat.cols, mat.rows, 0,
                           mat.ptr());
    } else {
        // We need to get it into the right format first.  Convert then
        // send
        cv::Mat floatedMat;
        mat.convertTo(floatedMat, CV_32F);

        return cl::Image2D(context, 
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), 
                           floatedMat.cols, floatedMat.rows, 0,
                           floatedMat.ptr());
    }
}


