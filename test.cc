#include <iostream>
#include <fstream>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

#include "filterer.h"

#include "cv.h"
#include "highgui.h"
#include <stdexcept>

struct dtcwtFilters {
    cl::Buffer level1h0;
    cl::Buffer level1h1;
    cl::Buffer level1hbp;

    cl::Buffer level2h0;
    cl::Buffer level2h1;
    cl::Buffer level2hbp;
};

struct dtcwtKernels {
    cl::Kernel colDecimateFilter;
    cl::Kernel rowDecimateFilter;
    cl::Kernel colFilter;
    cl::Kernel rowFilter;
    cl::Kernel quadToComplex;
    cl::Kernel cornernessMap;
};

dtcwtKernels getKernels(cl::Program& program)
{
    dtcwtKernels kernels;

    // Turn these into kernels
    kernels.rowDecimateFilter = cl::Kernel(program, "rowDecimateFilter");
    kernels.colDecimateFilter = cl::Kernel(program, "colDecimateFilter");
    kernels.rowFilter = cl::Kernel(program, "rowFilter");
    kernels.colFilter = cl::Kernel(program, "colFilter");
    kernels.quadToComplex = cl::Kernel(program, "quadToComplex");
    kernels.cornernessMap = cl::Kernel(program, "cornernessMap");

    return kernels;
}
    



void dtcwtTransform(cl::Context& context, cl::CommandQueue& commandQueue,
                    std::vector<std::vector<cl::Image2D> >& output,
                    cl::Image2D& input, dtcwtFilters& filters,
                    dtcwtKernels& kernels,
                    int numLevels, int startLevel = 1)
{
    //std::vector<std::vector<cl::Image2D> > output; 

    cl::Image2D lolo;

         /*   cv::Mat im = filterer.getImage2D(input);
            cv::imshow("Output", im);
            cv::waitKey();*/



    // Go down the tree until the point where we need to start recording
    // the results
    for (int n = 1; n < startLevel; ++n) {

        if (n == 1) {

            int width = input.getImageInfo<CL_IMAGE_WIDTH>();
            int height = input.getImageInfo<CL_IMAGE_HEIGHT>();

            // Pad if an odd number of pixels
            bool padW = width & 1,
                 padH = height & 1;

            // Low-pass filter the rows...
            cl::Image2D lo = 
                createImage2D(context, width + padW, height);
            rowFilter(context, commandQueue, kernels.rowFilter,
                      lo, input, filters.level1h0);

            // ...and the columns
            lolo = 
                createImage2D(context, width + padW, height + padH);
            colFilter(context, commandQueue, kernels.colFilter,
                      lolo, lo, filters.level1h0);

        } else {

            int width = lolo.getImageInfo<CL_IMAGE_WIDTH>();
            int height = lolo.getImageInfo<CL_IMAGE_HEIGHT>();

            // Pad if a non-multiple of four
            bool padW = (width % 4) != 0,
                 padH = (height % 4) != 0;

            // Low-pass filter the rows...
            cl::Image2D lo = 
                createImage2D(context, width / 2 + padW, height);
            rowDecimateFilter(context, commandQueue, kernels.colDecimateFilter,
                              lo, lolo, filters.level2h0, padW);

            // ...and the columns
            lolo = 
                createImage2D(context, width / 2 + padW, height / 2 + padH);
            colDecimateFilter(context, commandQueue, kernels.colDecimateFilter,
                              lolo, lo, filters.level2h0, padH);
            
        }

    }

    // Transform the image
    for (int n = startLevel; n < (startLevel + numLevels); ++n) {
        cl::Image2D hilo, lohi, bpbp;

        if (n == 1) {

            int width = input.getImageInfo<CL_IMAGE_WIDTH>();
            int height = input.getImageInfo<CL_IMAGE_HEIGHT>();

            // Pad if an odd number of pixels
            bool padW = width & 1,
                 padH = height & 1;

            // Low (row) - high (cols)
            cl::Image2D lo = 
                createImage2D(context, width + padW, height);
            rowFilter(context, commandQueue, kernels.rowFilter, 
                      lo, input, filters.level1h0);

            
            lohi =
                createImage2D(context, width + padW, height + padH);
            colFilter(context, commandQueue, kernels.colFilter, 
                      lohi, lo, filters.level1h1);

            // High (row) - low (cols)
            cl::Image2D hi =
                createImage2D(context, width + padW, height);
            rowFilter(context, commandQueue, kernels.rowFilter,
                      hi, input, filters.level1h1);

            hilo =
                createImage2D(context, width + padW, height + padH);
            colFilter(context, commandQueue, kernels.colFilter,
                      hilo, hi, filters.level1h0);

            // Band pass - band pass
            cl::Image2D bp =
                createImage2D(context, width + padW, height);
            rowFilter(context, commandQueue, kernels.rowFilter,
                      bp, input, filters.level1hbp);

            bpbp =
                createImage2D(context, width + padW, height + padH);
            colFilter(context, commandQueue, kernels.colFilter,
                      bpbp, bp, filters.level1hbp);


            // Low - low
            lolo = 
                createImage2D(context, width + padW, height + padH);
            colFilter(context, commandQueue, kernels.colFilter,
                      lolo, lo, filters.level1h0);



        } else {

            /*cv::Mat im = filterer.getImage2D(lolo);
            cv::imshow("Output", im);
            cv::waitKey();*/

            int width = lolo.getImageInfo<CL_IMAGE_WIDTH>();
            int height = lolo.getImageInfo<CL_IMAGE_HEIGHT>();

            // Pad if an odd number of pixels
            bool padW = (width % 4) != 0,
                 padH = (height % 4) != 0;

            // Low (row) - high (cols)
            cl::Image2D lo = 
                createImage2D(context, width / 2 + padW, height);
            rowDecimateFilter(context, commandQueue, kernels.rowDecimateFilter,
                              lo, lolo, filters.level2h0, padW);

            lohi =
                createImage2D(context, width / 2 + padW, height / 2 + padH);
            colDecimateFilter(context, commandQueue, kernels.colDecimateFilter,
                              lohi, lo, filters.level2h1, padH);


            // High (row) - low (cols)
            cl::Image2D hi =
                createImage2D(context, width / 2 + padW, height);
            rowDecimateFilter(context, commandQueue, kernels.rowDecimateFilter,
                              hi, lolo, filters.level2h1, padW);

            hilo =
                createImage2D(context, width / 2 + padW, height / 2 + padH);
            colDecimateFilter(context, commandQueue, kernels.colDecimateFilter,
                              hilo, hi, filters.level2h0, padH);


            // Band pass - band pass
            cl::Image2D bp =
                createImage2D(context, width / 2 + padW, height);
            rowDecimateFilter(context, commandQueue, kernels.rowDecimateFilter,
                              bp, lolo, filters.level2hbp, padW);

            bpbp =
                createImage2D(context, width / 2 + padW, height / 2 + padH);
            colDecimateFilter(context, commandQueue, kernels.colDecimateFilter,
                              bpbp, bp, filters.level2hbp, padH);


            // Low - low
            lolo = 
                createImage2D(context, width / 2 + padW, height / 2 + padH);
            colDecimateFilter(context, commandQueue, kernels.colDecimateFilter,
                              lolo, lo, filters.level2h0, padH);

        }

        output.push_back(std::vector<cl::Image2D>());

        int idx = n - startLevel;
        int width = hilo.getImageInfo<CL_IMAGE_WIDTH>() / 2;
        int height = hilo.getImageInfo<CL_IMAGE_HEIGHT>() / 2;

        for (int n = 0; n < 12; ++n)
            output[idx].push_back(createImage2D(context, width, height));


        quadToComplex(context, commandQueue, kernels.quadToComplex,
                      output[idx][2], output[idx][2+6],
                      output[idx][3], output[idx][3+6],
                      lohi);

        quadToComplex(context, commandQueue, kernels.quadToComplex,
                      output[idx][0], output[idx][0+6],
                      output[idx][5], output[idx][5+6],
                      hilo);

        quadToComplex(context, commandQueue, kernels.quadToComplex,
                      output[idx][4], output[idx][4+6],
                      output[idx][1], output[idx][1+6],
                      bpbp);

    }
    //return output;

}


dtcwtFilters createFilters(cl::Context& context,
                           cl::CommandQueue& commandQueue)
{
    const float level1h0[13] = {
       -0.0018,
             0,
        0.0223,
       -0.0469,
       -0.0482,
        0.2969,
        0.5555,
        0.2969,
       -0.0482,
       -0.0469,
        0.0223,
             0,
       -0.0018
    };

    const float level1h1[19] = {
       -0.0001,
             0,
        0.0013,
       -0.0019,
       -0.0072,
        0.0239,
        0.0556,
       -0.0517,
       -0.2998,
        0.5594,
       -0.2998,
       -0.0517,
        0.0556,
        0.0239,
       -0.0072,
       -0.0019,
        0.0013,
             0,
       -0.0001
    };

    const float level1hbp[19] = {
       -0.0004,
       -0.0006,
       -0.0001,
        0.0042,
        0.0082,
       -0.0074,
       -0.0615,
       -0.1482,
       -0.1171,
        0.6529,
       -0.1171,
       -0.1482,
       -0.0615,
       -0.0074,
        0.0082,
        0.0042,
       -0.0001,
       -0.0006,
       -0.0004
    };


    const float level2h0[14] = {
       -0.0046,
       -0.0054,
        0.0170,
        0.0238,
       -0.1067,
        0.0119,
        0.5688,
        0.7561,
        0.2753,
       -0.1172,
       -0.0389,
        0.0347,
       -0.0039,
        0.0033
    };

    const float level2h1[14] = {
       -0.0033,
       -0.0039,
       -0.0347,
       -0.0389,
        0.1172,
        0.2753,
       -0.7561,
        0.5688,
       -0.0119,
       -0.1067,
       -0.0238,
        0.0170,
        0.0054,
       -0.0046
    };

    const float level2hbp[14] = {
       -0.0028,
       -0.0004,
        0.0210,
        0.0614,
        0.1732,
       -0.0448,
       -0.8381,
        0.4368,
        0.2627,
       -0.0076,
       -0.0264,
       -0.0255,
       -0.0096,
       -0.0000
    };

    dtcwtFilters filters;
    filters.level1h0 = createBuffer(context, commandQueue, level1h0, 13);
    filters.level1h1 = createBuffer(context, commandQueue, level1h1, 19);
    filters.level1hbp = createBuffer(context, commandQueue, level1hbp, 19);

    filters.level2h0 = createBuffer(context, commandQueue, level2h0, 14);
    filters.level2h1 = createBuffer(context, commandQueue, level2h1, 14);
    filters.level2hbp = createBuffer(context, commandQueue, level2hbp, 14);

    return filters;
}


int main()
{
    try {
         // Retrive platform information
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.size() == 0)
            throw std::runtime_error("No platforms!");

        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // Create a context to work in 
        cl::Context context(devices);


        // Open the program, find its length and read it out
        std::ifstream sourceFile("kernel.cl", std::ios::in | std::ios::ate);
        std::string kernelSource(sourceFile.tellg(), ' ');
        sourceFile.seekg(0, std::ios::beg);
        sourceFile.read(&kernelSource[0], kernelSource.length());

        // Create a program compiled from the source code (read in previously)
        cl::Program::Sources source;
        source.push_back(std::pair<const char*, size_t>(kernelSource.c_str(),
                                    kernelSource.length()));
        cl::Program program(context, source);
        try {
            program.build(devices);
        } catch(cl::Error err) {
            std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) 
                      << std::endl;
            throw;
        }

        // Ready the command queue on the first device to hand
        cl::CommandQueue commandQueue(context, devices[0]);

        std::string displays[] = {"S1", "S2", "S3", "S4", "S5", "S6"};

        // Read the image (forcing to RGB), and convert it to floats ("1")
        //cv::Mat inImage = cv::imread("~/Download/PVTRA101a01/00003449.jpg", 1);
        cv::Mat inImage = cv::imread("traffic.bmp", 1);

        // Open the camera for reading
///*
        cv::VideoCapture videoIn(0);
        videoIn.set(CV_CAP_PROP_FRAME_WIDTH, 320);
        videoIn.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

//*/

        dtcwtFilters filters = createFilters(context, commandQueue);
        dtcwtKernels kernels = getKernels(program);
///*
        for (int n = 0; n < 6; ++n)
            cv::namedWindow(displays[n], CV_WINDOW_NORMAL);
        cv::namedWindow("Cornerness", CV_WINDOW_NORMAL);
//*/ 
        cv::Mat vidImage;
        cv::Mat outImage;

        int x = 0;
        int numLevels = 4;
///*    
        while (1) {
            videoIn >> vidImage;
//*/
            vidImage = inImage;

            cv::Mat inputTmp;
            cv::Mat inputTmp2;
            vidImage.convertTo(inputTmp, CV_32F);
            cvtColor(inputTmp, inputTmp2, CV_RGB2GRAY);
            cv::Mat input(vidImage.size(), CV_32FC4);

            // Working in BGRA (for ref.)
            // Now, put it into 4 channels
            int fromTo[] = {0,0, 0,1, 0,2, -1,3};
            cv::mixChannels(&inputTmp2, 1, &input, 1, fromTo, 4);

            // Send to the graphics card
            cl::Image2D img(createImage2D(context, input));

            // Do the calculations there
            std::vector<std::vector<cl::Image2D> > results;
            dtcwtTransform(context, commandQueue,
                           results, img, filters, kernels,
                           numLevels, 2);

            const int l = 0;
            int width = results[l][0].getImageInfo<CL_IMAGE_WIDTH>();
            int height = results[l][0].getImageInfo<CL_IMAGE_HEIGHT>();

            cv::Mat disp(height, width, CV_32FC4);

            // Read them out
///*
            for (int n = 0; n < 6; ++n) {
                cv::Mat re = getImage2D(commandQueue, results[l][n]);
                cv::Mat im = getImage2D(commandQueue, results[l][n+6]);
                //cv::Mat outArea = disp.colRange(n*width, (n+1)*width-1);
                cv::sqrt(re.mul(re) + im.mul(im), disp);
                cv::imshow(displays[n], disp / 64.0f);
            }
//*/
            cl::Image2D mapImg;
            cornernessMap(context, commandQueue, kernels.cornernessMap, 
                          mapImg, results[l]);
///*
            cv::Mat map = getImage2D(commandQueue, mapImg);
            cv::imshow("Cornerness", map / 32.0f);

            // Display
            std::cout << "Displayed! " << x++ <<  std::endl;
            cv::waitKey(1);
//*/
///*    
        }
//*/
/*
        for (int n = 0; n < numLevels; ++n) {
            for (int m = 0; m < 6; ++m) {
                cv::Mat re = filterer.getImage2D(results[n][m]);
                cv::Mat im = filterer.getImage2D(results[n][m+6]);

                //cv::imshow("Output", filteredImage + 0.5f);
            
                cv::imshow("Output", re);
                cv::waitKey();

                cv::imshow("Output", im);
                cv::waitKey();
                cv::imwrite("out.bmp", re);

            }
        }
*/
    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }

                     
    return 0;
}



