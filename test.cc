#include <iostream>
#include <fstream>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

#include "filterer.h"

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
/*
            cv::Mat im = getImage2D(commandQueue, input);
            cv::imshow("Output", im);
            cv::waitKey();
*/


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
                createImage2D(context, commandQueue, width + padW, height);
            //rowFilter(context, commandQueue, kernels.rowFilter,
            //          lo, input, filters.level1h0);

    std::cout << "Hello" << std::endl;
            // ...and the columns
            lolo = 
                createImage2D(context, commandQueue, width + padW, height + padH);
            colFilter(context, commandQueue, kernels.colFilter,
                      lolo, lo, filters.level1h0);
            std::cout << "Hello 2" << std::endl;

        } else {

            int width = lolo.getImageInfo<CL_IMAGE_WIDTH>();
            int height = lolo.getImageInfo<CL_IMAGE_HEIGHT>();

            // Pad if a non-multiple of four
            bool padW = (width % 4) != 0,
                 padH = (height % 4) != 0;

            // Low-pass filter the rows...
            cl::Image2D lo = 
                createImage2D(context, commandQueue, width / 2 + padW, height);
            rowDecimateFilter(context, commandQueue, kernels.colDecimateFilter,
                              lo, lolo, filters.level2h0, padW);

            // ...and the columns
            lolo = 
                createImage2D(context, commandQueue, width / 2 + padW, height / 2 + padH);
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
                createImage2D(context, commandQueue, width + padW, height);
            //rowFilter(context, commandQueue, kernels.rowFilter, 
            //          lo, input, filters.level1h0);

            
            lohi =
                createImage2D(context, commandQueue, width + padW, height + padH);
            colFilter(context, commandQueue, kernels.colFilter, 
                      lohi, lo, filters.level1h1);

            // High (row) - low (cols)
            cl::Image2D hi =
                createImage2D(context, commandQueue, width + padW, height);
            //rowFilter(context, commandQueue, kernels.rowFilter,
            //          hi, input, filters.level1h1);

            hilo =
                createImage2D(context, commandQueue, width + padW, height + padH);
            colFilter(context, commandQueue, kernels.colFilter,
                      hilo, hi, filters.level1h0);

            // Band pass - band pass
            cl::Image2D bp =
                createImage2D(context, commandQueue, width + padW, height);
            //rowFilter(context, commandQueue, kernels.rowFilter,
            //          bp, input, filters.level1hbp);

            bpbp =
                createImage2D(context, commandQueue, width + padW, height + padH);
            colFilter(context, commandQueue, kernels.colFilter,
                      bpbp, bp, filters.level1hbp);


            // Low - low
            lolo = 
                createImage2D(context, commandQueue, width + padW, height + padH);
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
                createImage2D(context, commandQueue, width / 2 + padW, height);
            rowDecimateFilter(context, commandQueue, kernels.rowDecimateFilter,
                              lo, lolo, filters.level2h0, padW);

            lohi =
                createImage2D(context, commandQueue, width / 2 + padW, height / 2 + padH);
            colDecimateFilter(context, commandQueue, kernels.colDecimateFilter,
                              lohi, lo, filters.level2h1, padH);


            // High (row) - low (cols)
            cl::Image2D hi =
                createImage2D(context, commandQueue, width / 2 + padW, height);
            rowDecimateFilter(context, commandQueue, kernels.rowDecimateFilter,
                              hi, lolo, filters.level2h1, padW);

            hilo =
                createImage2D(context, commandQueue, width / 2 + padW, height / 2 + padH);
            colDecimateFilter(context, commandQueue, kernels.colDecimateFilter,
                              hilo, hi, filters.level2h0, padH);


            // Band pass - band pass
            cl::Image2D bp =
                createImage2D(context, commandQueue, width / 2 + padW, height);
            rowDecimateFilter(context, commandQueue, kernels.rowDecimateFilter,
                              bp, lolo, filters.level2hbp, padW);

            bpbp =
                createImage2D(context, commandQueue, width / 2 + padW, height / 2 + padH);
            colDecimateFilter(context, commandQueue, kernels.colDecimateFilter,
                              bpbp, bp, filters.level2hbp, padH);


            // Low - low
            lolo = 
                createImage2D(context, commandQueue, width / 2 + padW, height / 2 + padH);
            colDecimateFilter(context, commandQueue, kernels.colDecimateFilter,
                              lolo, lo, filters.level2h0, padH);

        }

        output.push_back(std::vector<cl::Image2D>());

        int idx = n - startLevel;
        int width = hilo.getImageInfo<CL_IMAGE_WIDTH>() / 2;
        int height = hilo.getImageInfo<CL_IMAGE_HEIGHT>() / 2;

        for (int n = 0; n < 12; ++n)
            output[idx].push_back(createImage2D(context, commandQueue, width, height));


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


        // Ready the command queue on the first device to hand
        cl::CommandQueue commandQueue(context, devices[0]);

        dtcwtFilters filters = createFilters(context, commandQueue);

        int x = 0;
        int numLevels = 4;

        //-----------------------------------------------------------------
        // Starting test code
        const size_t width = 5, height = 10;
  
        const cl::ImageFormat imageFormat(CL_LUMINANCE, CL_FLOAT);
        cl::Image2D inImage(context, 0, imageFormat, width, height);
        cl::Image2D outImage(context, 0, imageFormat, width, height);


        cl::size_t<3> origin, extents;
        origin.push_back(0);
        origin.push_back(0);
        origin.push_back(0);
        extents.push_back(width);
        extents.push_back(height);
        extents.push_back(1);

        float input[height][width] = {0.0f};
        input[3][2] = 1.0f;

        commandQueue.enqueueWriteImage(inImage, CL_TRUE, origin, extents,
                                       0, 0, &input[0][0]);


        ColFilter colFilter(context, devices);
        
        std::vector<cl::Event> waitEvents(1);

        colFilter(commandQueue, outImage, inImage, filters.level1h0,
                  0, &waitEvents[0]);

        float output[height][width];

        commandQueue.enqueueReadImage(outImage, CL_TRUE, origin, extents,
                                      0, 0, &output[0][0]);

        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x)
                std::cout << output[y][x] << "\t";

            std::cout << std::endl;
        }







        //-----------------------------------------------------------------

#if 0
///*    
        while (1) {
            videoIn >> vidImage;
//*/
            vidImage = inImage;

            cv::Mat inputTmp;
            vidImage.convertTo(inputTmp, CV_32F);

            cv::Mat inputImg;
            cvtColor(inputTmp, inputImg, CV_RGB2GRAY);
            //cv::Mat input(vidImage.size(), CV_32FC4);

            // Working in BGRA (for ref.)
            // Now, put it into 4 channels
            //int fromTo[] = {0,0, 0,1, 0,2, -1,3};
            //cv::mixChannels(&inputTmp2, 1, &input, 1, fromTo, 4);

            // Send to the graphics card
            cl::Image2D img(createImage2D(context, commandQueue, inputImg));

            // Do the calculations there
            std::vector<std::vector<cl::Image2D> > results;
            dtcwtTransform(context, commandQueue,
                           results, img, filters, kernels,
                           numLevels, 2);

            const int l = 0;
            int width = results[l][0].getImageInfo<CL_IMAGE_WIDTH>();
            int height = results[l][0].getImageInfo<CL_IMAGE_HEIGHT>();

            cv::Mat disp(height, width, CV_32FC1);

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

            cv::waitKey(1);
            
            // Display
            std::cout << "Displayed! " << x++ <<  std::endl;

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

#endif
    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
                     
    return 0;
}



