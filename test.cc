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


int main()
{
    try {

        cl::Platform platform;
        std::vector<cl::Device> devices;
        cl::Context context;
        cl::CommandQueue commandQueue; 
        std::tie(platform, devices, context, commandQueue) = initOpenCL();

        const int numLevels = 6;
        const int startLevel = 1;


        //-----------------------------------------------------------------
        // Starting test code
  
        // Read in image
        cv::Mat bmp = cv::imread("test.bmp", 0);
        cl::Image2D inImage = createImage2D(context, bmp);

        std::cout << bmp.rows << " " << bmp.cols << std::endl;
        std::cout << "Creating Dtcwt" << std::endl;

        Dtcwt dtcwt(context, devices, commandQueue);

        std::cout << "Creating the DTCWT environment..." << std::endl;

        DtcwtTemps env = dtcwt.createContext(bmp.cols, bmp.rows,
                                           numLevels, startLevel);

        std::cout << "Creating the subband output images..." << std::endl;
        DtcwtOutput out(env);

        std::cout << "Running DTCWT" << std::endl;



        time_t start, end;
        const int numFrames = 1000;
        time(&start);
            for (int n = 0; n < numFrames; ++n) {
                dtcwt(commandQueue, inImage, env, out);
                commandQueue.finish();
            }
        time(&end);
        std::cout << (numFrames / difftime(end, start))
		  << " fps" << std::endl;
        std::cout << numFrames << " frames in " 
                  << difftime(end, start) << "s" << std::endl;

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



