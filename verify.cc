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

#include <cv.h>
#include <highgui.h>


std::tuple<cl::Platform, std::vector<cl::Device>, 
           cl::Context, cl::CommandQueue> 
    initOpenCL();

cl::Image2D createImage2D(cl::Context& context, cv::Mat& mat);

std::tuple<Filters, Filters>
        createFilters(cl::Context& context, cl::CommandQueue& commandQueue)
{
    Filters level1, level2;

    level1.h0 = createBuffer(context, commandQueue,
           { -0.0018, 0, 0.0223, -0.0469, -0.0482, 0.2969, 0.5555, 0.2969,
             -0.0482, -0.0469, 0.0223, 0, -0.0018} );

    level1.h1 = createBuffer(context, commandQueue, 
           { -0.0001, 0, 0.0013, -0.0019, -0.0072, 0.0239, 0.0556, -0.0517,
             -0.2998, 0.5594, -0.2998, -0.0517, 0.0556, 0.0239, -0.0072,
             -0.0019, 0.0013, 0, -0.0001 } );
    
    level1.hbp = createBuffer(context, commandQueue, 
           { -0.0004, -0.0006, -0.0001, 0.0042, 0.0082, -0.0074, -0.0615,
             -0.1482, -0.1171, 0.6529, -0.1171, -0.1482, -0.0615, -0.0074, 
             0.0082, 0.0042, -0.0001, -0.0006, -0.0004 } );

    level2.h0 = createBuffer(context, commandQueue, 
           { -0.0046, -0.0054, 0.0170, 0.0238, -0.1067, 0.0119, 0.5688,
             0.7561, 0.2753, -0.1172, -0.0389, 0.0347, -0.0039, 0.0033 } );

    level2.h1 = createBuffer(context, commandQueue, 
           { -0.0033, -0.0039, -0.0347, -0.0389, 0.1172, 0.2753, -0.7561,
             0.5688, -0.0119, -0.1067, -0.0238, 0.0170, 0.0054, -0.0046 } );

    level2.hbp = createBuffer(context, commandQueue, 
           { -0.0028, -0.0004, 0.0210, 0.0614, 0.1732, -0.0448, -0.8381,
             0.4368, 0.2627, -0.0076, -0.0264, -0.0255, -0.0096, -0.0000 } );

    return std::make_tuple(level1, level2);
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
            out << output[y][x][0]
                << (output[y][x] >= 0? "+" : "")
                << output[y][x][1] << "j"
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
        const int startLevel = 1;


        //-----------------------------------------------------------------
        // Starting test code
  
        // Read in image
        cv::Mat bmp = cv::imread("test.bmp", 0);
        cl::Image2D inImage = createImage2D(context, bmp);

        std::cout << bmp.rows << " " << bmp.cols << std::endl;
        std::cout << "Creating Dtcwt" << std::endl;
        Dtcwt dtcwt(context, devices);

        Filters level1, level2;
        std::tie(level1, level2) = createFilters(context, commandQueue);

        DtcwtContext env = dtcwt.createContext(bmp.cols, bmp.rows,
                                               numLevels, startLevel,
                                               level1, level2);

        std::cout << "Running DTCWT" << std::endl;

        dtcwt(commandQueue, inImage, env);
        commandQueue.finish();

        std::cout << "Saving image" << std::endl;

        saveComplexImage("sb0.dat", commandQueue, env.outputs[0][0]);
        saveComplexImage("sb1.dat", commandQueue, env.outputs[0][1]);
        saveComplexImage("sb2.dat", commandQueue, env.outputs[0][2]);
        saveComplexImage("sb3.dat", commandQueue, env.outputs[0][3]);
        saveComplexImage("sb4.dat", commandQueue, env.outputs[0][4]);
        saveComplexImage("sb5.dat", commandQueue, env.outputs[0][5]);

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


