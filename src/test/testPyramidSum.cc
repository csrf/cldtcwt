#include <iostream>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "util/clUtil.h"


#include "KeypointDetector/EnergyMaps/PyramidSum/pyramidSum.h"




int main()
{
    try {

        CLContext context;


        // Ready the command queue on the first device to hand
        cl::CommandQueue cq(context.context, context.devices[0]);

        //-----------------------------------------------------------------
        // Starting test code
        

        PyramidSum pyramidSum(context.context, context.devices);

        // Number of floats in each position

        const int width = 6, height = 6;
        // Set up data for the input image

        std::vector<float> in1V =
            {0, 1, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0};

        std::vector<float> in2V =
            {0, 0, 0,
             0, 1, 0,
             0, 0, 0};

        cl::Image2D in1 = {
            context.context, 
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), 
            width, height, 0,
            &in1V[0]
        };

        cl::Image2D in2 = {
            context.context, 
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), 
            width / 2, height / 2, 0,
            &in2V[0]
        };

        cl::Image2D output = {
            context.context, 
            CL_MEM_READ_WRITE,
            cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), 
            width, height, 0,
        };

        pyramidSum(cq, in1, 1.0, in2, 1.0, output);
       
        std::vector<float> outputV(width*height);

        cq.enqueueReadImage(output, CL_TRUE, 
                            makeCLSizeT<3>({0,0,0}), 
                            makeCLSizeT<3>({width, height, 1}), 
                            0, 0, &outputV[0]);

        // Display output
        int pos = 0;
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y)
                std::cout << outputV[pos++] << "\t";
            std::cout << "\n";
        }

    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
                     
    return 0;
}


