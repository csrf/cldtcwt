#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

#include "filterer.h"
#include "clUtil.h"
#include "dtcwt.h"
#include <iomanip>

#include <ctime>

#include <stdexcept>

#include <highgui.h>


std::tuple<cl::Platform, std::vector<cl::Device>, 
           cl::Context, cl::CommandQueue> 
    initOpenCL();


void displayRealImage(cl::CommandQueue& cq, cl::Image2D& image)
{
    const size_t width = image.getImageInfo<CL_IMAGE_WIDTH>(),
                height = image.getImageInfo<CL_IMAGE_HEIGHT>();
    float output[height][width];
    readImage2D(cq, &output[0][0], image);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x)
            std::cout << output[y][x] << "\t"; 

        std::cout << std::endl;
    }

    std::cout << std::endl;
}


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
  
        cv::Mat input = cv::Mat::zeros(32, 4, cv::DataType<float>::type);
        input.at<float>(16,3) = 1.0f;
        cl::Image2D inImage = createImage2D(context, input);

        Filter h = { 
            context, devices, 
            createBuffer(context, commandQueue, {0.5, 1, 0.5}),
            Filter::y 
        };

        cl::Image2D outImage
            = createImage2D(context, inImage.getImageInfo<CL_IMAGE_WIDTH>(),
                                     inImage.getImageInfo<CL_IMAGE_HEIGHT>());

        h(commandQueue, inImage, outImage);


        DecimateFilter hd = { 
            context, devices, 
            createBuffer(context, commandQueue, {0.5, 0.0, 1.0, 0.5}),
            DecimateFilter::y 
        };

        cl::Image2D outImageD
            = createImage2D(context, inImage.getImageInfo<CL_IMAGE_WIDTH>(),
                                     inImage.getImageInfo<CL_IMAGE_HEIGHT>() / 2);

        hd(commandQueue, inImage, outImageD);

        commandQueue.finish();

        displayRealImage(commandQueue, outImage);
        displayRealImage(commandQueue, outImageD);

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
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    // Create a context to work in 
    cl::Context context(devices);

    // Ready the command queue on the first device to hand
    cl::CommandQueue commandQueue(context, devices[0]);

    return std::make_tuple(platforms[0], devices, context, commandQueue);
}



