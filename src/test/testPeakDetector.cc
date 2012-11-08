#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "util/clUtil.h"
#include <iomanip>

#include <stdexcept>



#include "KeypointDetector/peakDetector.h"




int main()
{
    try {

        CLContext context;


        // Ready the command queue on the first device to hand
        cl::CommandQueue cq(context.context, context.devices[0]);

        //-----------------------------------------------------------------
        // Starting test code
        
        PeakDetector peakDetector(context.context, context.devices);

        PeakDetectorResults results
            = peakDetector.createResultsStructure({20}, 20);

        const size_t posLen = 4;
        // Number of floats in each position

        const int width = 20, height = 20;
        // Set up data for the input image
        float data[height][width];
        for (int x = 0; x < width; ++x)
            for (int y = 0; y < height; ++y)
                data[y][x] = 0.0f;
        data[10][5] = 1.0f;
        data[13][12] = 1.0f;
        data[14][12] = 2.0f;
        data[1][1] = 1.0f;


        cl::Image2D inImage = {
            context.context, 
            CL_MEM_READ_WRITE,
            cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), 
            width, height, 0
        };


        writeImage2D(cq, inImage, &data[0][0]);
        cq.finish();

        std::cout << "Running" << std::endl;
                                       
        peakDetector(cq, {&inImage}, {1.0f}, 0.1f,
                         results);

        cq.finish();
        std::cout << "Done" << std::endl;

        int numOutputsVal;
        cq.enqueueReadBuffer(results.cumCounts, CL_TRUE, 
                             sizeof(cl_uint), sizeof(cl_uint),
                             &numOutputsVal);

        std::cout << numOutputsVal << " outputs" << std::endl;

        /*numOutputsVal = (numOutputsVal > 10) ? 10 : numOutputsVal;
        
        if (numOutputsVal > 0) {
            std::vector<float> results(numOutputsVal * posLen);
            cq.enqueueReadBuffer(outputs, true, 0, 
                                           numOutputsVal * posLen * sizeof(float),
                                           &results[0]);

            for (auto v: results)
                std::cout << v << std::endl;
        }*/


    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
                     
    return 0;
}


