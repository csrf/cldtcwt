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

cl::Image2D createImage2D(cl::Context& context, cv::Mat& mat);

std::tuple<Filters, Filters>
        createFilters(cl::Context& context, cl::CommandQueue& commandQueue)
{
    Filters level1, level2;

    level1.h0 = createBuffer(context, commandQueue, { 
          -0.001757812500000,
           0.000000000000000,
           0.022265625000000,
          -0.046875000000000,
          -0.048242187500000,
           0.296875000000000,
           0.555468750000000,
           0.296875000000000,
          -0.048242187500000,
          -0.046875000000000,
           0.022265625000000,
           0.000000000000000,
          -0.001757812500000
    });

    level1.h1 = createBuffer(context, commandQueue, { 
          -0.000070626395089,
           0.000000000000000,
           0.001341901506696,
          -0.001883370535714,
          -0.007156808035714,
           0.023856026785714,
           0.055643136160714,
          -0.051688058035714,
          -0.299757603236607,
           0.559430803571429,
          -0.299757603236607,
          -0.051688058035714,
           0.055643136160714,
           0.023856026785714,
          -0.007156808035714,
          -0.001883370535714,
           0.001341901506696,
           0.000000000000000,
          -0.000070626395089
    } );
    
    level1.hbp = createBuffer(context, commandQueue, { 
          -3.68250025673202e-05,
          -6.22253585579744e-04,
          -7.81782479825950e-05,
           4.18582084706810e-03,
           8.19178717888364e-03,
          -7.42327402480263e-03,
          -6.15384268799117e-02,
          -1.48158230911691e-01,
          -1.17076301639216e-01,
           6.52908215843590e-01,
          -1.17076301639216e-01,
          -1.48158230911691e-01,
          -6.15384268799117e-02,
          -7.42327402480263e-03,
           8.19178717888364e-03,
           4.18582084706810e-03,
          -7.81782479825949e-05,
          -6.22253585579744e-04,
          -3.68250025673202e-04
    } );

    level2.h0 = createBuffer(context, commandQueue, {
          -0.00455689562847549,
          -0.00543947593727412,
           0.01702522388155399,
           0.02382538479492030,
          -0.10671180468666540,
           0.01186609203379700,
           0.56881042071212273,
           0.75614564389252248,
           0.27529538466888204,
          -0.11720388769911527,
          -0.03887280126882779,
           0.03466034684485349,
          -0.00388321199915849,
           0.00325314276365318
    } );

    level2.h1 = createBuffer(context, commandQueue, {
          -0.00325314276365318,
          -0.00388321199915849,
          -0.03466034684485349,
          -0.03887280126882779,
           0.11720388769911527,
           0.27529538466888204,
          -0.75614564389252248,
           0.56881042071212273,
          -0.01186609203379700,
          -0.10671180468666540,
          -0.02382538479492030,
           0.01702522388155399,
           0.00543947593727412,
          -0.00455689562847549
    } );

    level2.hbp = createBuffer(context, commandQueue, {
          -2.77165349347537e-03,
          -4.32919303381105e-04,
           2.10100577283097e-02,
           6.14446533755929e-02,
           1.73241472867428e-01,
          -4.47647940175083e-02,
          -8.38137840090472e-01,
           4.36787385780317e-01,
           2.62691880616686e-01,
          -7.62474758151248e-03,
          -2.63685613793659e-02,
          -2.54554351814246e-02,
          -9.59514305416110e-03,
          -2.43562670333119e-05
    } );

    return std::make_tuple(level1, level2);
}


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

        // This is the input we really want to put into a decimating layer
        cv::Mat input = cv::Mat::zeros(64, 64, cv::DataType<float>::type);
        input.at<float>(30,30) = 1.0f;
  
        //cl::Image2D baseInImage = createImage2D(context, baseInput);
        cl::Image2D inImage = createImage2D(context, input);

        std::cout << "Creating Dtcwt" << std::endl;

        Filters level1, level2;
        std::tie(level1, level2) = createFilters(context, commandQueue);

        Dtcwt dtcwt(context, devices, level1, level2);

        DtcwtTemps env = dtcwt.createContext(64, 64,
                                             numLevels, startLevel);

        DtcwtOutput sbOutputs = {env};

        std::cout << "Running DTCWT" << std::endl;

        
        dtcwt.decimateFilter(commandQueue, inImage, {}, 
                             env.levelTemps[1], &sbOutputs.subbands[1]);
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


