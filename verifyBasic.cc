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



void saveRealImage(std::string filename,
                   cl::CommandQueue& cq, cl::Image2D& image)
{
    const size_t width = image.getImageInfo<CL_IMAGE_WIDTH>(),
                height = image.getImageInfo<CL_IMAGE_HEIGHT>();
    float output[height][width];
    readImage2D(cq, &output[0][0], image);

    // Open the file for output
    std::ofstream out(filename, std::ios_base::trunc | std::ios_base::out);

    // Produce the output in a file readable by MATLAB dlmread
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            out << output[y][x] << ((x+1) < width? "," : "");
        }

        if ((y+1) < height)
            out << "\n";
    }
}



void saveComplexImage(std::string filename,
                      cl::CommandQueue& cq, cl::Image2D& image)
{
    const size_t width = image.getImageInfo<CL_IMAGE_WIDTH>(),
                height = image.getImageInfo<CL_IMAGE_HEIGHT>();
    float output[height][width][2];
    readImage2D(cq, &output[0][0][0], image);

    // Open the file for output
    std::ofstream out(filename, std::ios_base::trunc | std::ios_base::out);

    // Produce the output in a file readable by MATLAB dlmread
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            out << output[y][x][0];
            if (output[y][x][1] >= 0)
                out << "+";
            out << output[y][x][1] << "j"
                << ((x+1) < width? "," : "");
        }

        if ((y+1) < height)
            out << "\n";
    }
}


int main()
{
    try {

        cl::Platform platform;
        std::vector<cl::Device> devices;
        cl::Context context;
        cl::CommandQueue commandQueue; 
        std::tie(platform, devices, context, commandQueue) = initOpenCL();
const int numLevels = 6;
        const int startLevel = 0;

        //-----------------------------------------------------------------
        // Starting test code

        cv::Mat input = cv::Mat::zeros(128, 128, cv::DataType<float>::type);
        input.at<float>(63,63) = 1.0f;
  
        cl::Image2D inImage = createImage2D(context, input);

        std::cout << "Creating Dtcwt" << std::endl;


        Dtcwt dtcwt(context, devices, commandQueue);

        DtcwtTemps env = dtcwt.createContext(input.cols, input.rows,
                                             numLevels, startLevel);

        DtcwtOutput sbOutputs = {env};

        std::cout << "Running DTCWT" << std::endl;

        
        dtcwt(commandQueue, inImage, env, sbOutputs);
        commandQueue.finish();

        std::cout << "Saving image" << std::endl;

        saveRealImage("lolo2.dat", commandQueue, env.levelTemps[1].lolo);
        saveRealImage("lox.dat", commandQueue, env.levelTemps[1].lox);
        saveRealImage("lohi.dat", commandQueue, env.levelTemps[1].lohi);
        saveRealImage("hilo.dat", commandQueue, env.levelTemps[1].hilo);
        saveRealImage("xbp.dat", commandQueue, env.levelTemps[1].xbp);
        saveRealImage("bpbp.dat", commandQueue, env.levelTemps[1].bpbp);
        saveComplexImage("sb0.dat", commandQueue, sbOutputs.subbands[1].sb[0]);
        saveComplexImage("sb1.dat", commandQueue, sbOutputs.subbands[1].sb[1]);
        saveComplexImage("sb2.dat", commandQueue, sbOutputs.subbands[1].sb[2]);
        saveComplexImage("sb3.dat", commandQueue, sbOutputs.subbands[1].sb[3]);
        saveComplexImage("sb4.dat", commandQueue, sbOutputs.subbands[1].sb[4]);
        saveComplexImage("sb5.dat", commandQueue, sbOutputs.subbands[1].sb[5]);

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


