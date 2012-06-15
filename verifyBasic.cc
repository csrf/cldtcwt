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
#include <sstream>


std::tuple<cl::Platform, std::vector<cl::Device>, 
           cl::Context, cl::CommandQueue> 
    initOpenCL();


int main(int argc, char** argv)
{
    // Make sure we were given the image name to process
    if (argc < 2) {
        std::cerr << "No image provided!" << std::endl;
        return -1;
    }

    std::string filename(argv[1]);

    try {

        cl::Platform platform;
        std::vector<cl::Device> devices;
        cl::Context context;
        cl::CommandQueue commandQueue; 
        std::tie(platform, devices, context, commandQueue) = initOpenCL();

        // This should make sure all code gets exercised, including the
        // lolo decimated
        const int numLevels = 3;
        const int startLevel = 0;

        // Read the image in
        cv::Mat bmp = cv::imread(filename, 0) / 255.0f;
        cl::Image2D inImage = createImage2D(context, bmp);

        // Create the DTCWT itself
        Dtcwt dtcwt(context, devices, commandQueue);

        // Create the intermediate storage for the DTCWT
        DtcwtTemps env = dtcwt.createContext(bmp.cols, bmp.rows,
                                             numLevels, startLevel);

        // Create the outputs storage for the DTCWT
        DtcwtOutput sbOutputs = {env};

        // Perform DTCWT
        dtcwt(commandQueue, inImage, env, sbOutputs);
        commandQueue.finish();

        // Produce the outputs
        for (int l = 0; l < numLevels; ++l) {
            for (int sb = 0; sb < 6; ++sb) {

                // Construct output name in format 
                // <original name>.<level>.<subband number>
                std::ostringstream ss;
                ss << filename << "." << l << "." << sb;

                saveComplexImage(ss.str(), 
                                 commandQueue, 
                                 sbOutputs.subbands[l].sb[sb]);
            }
        }

        saveRealImage("in.dat", commandQueue, inImage);
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


