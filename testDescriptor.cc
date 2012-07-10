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

#include <sys/timeb.h>

#include <stdexcept>

#include <highgui.h>
#include "extractDescriptors.h"


int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Provide an image filename" << std::endl;
        return -1;
    }

    try {

        const int numLevels = 6;
        const int startLevel = 1;

        CLContext context;

        // Ready the command queue on the first device to hand
        cl::CommandQueue cq(context.context, context.devices[0]);
        Dtcwt dtcwt(context.context, context.devices, cq);

        // Ready the keypoint extractor
        DescriptorExtracter
            descriptorExtracter(context.context, context.devices, cq,
                                {{0,0}}, 1.f,
                                1, 0,
                                0);

        // Read in image
        cv::Mat bmp = cv::imread(argv[1], 0);
        cl::Image2D inImage = createImage2D(context.context, bmp);

        // Create temporaries and outputs for the DTCWT
        DtcwtTemps env = dtcwt.createContext(bmp.cols, bmp.rows,
                                           numLevels, startLevel);
        DtcwtOutput out(env);

        // Create locations to sample at
        cl::Buffer kplocs = createBuffer(context.context, cq, 
                                         {-0.5f, -0.5f});

        cl::Buffer output = createBuffer(context.context, cq, 
                                         {0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                                          0.f, 0.f, 0.f, 0.f, 0.f, 0.f});

        // Perform the transform
        dtcwt(cq, inImage, env, out);

        // Extract descriptors
        descriptorExtracter(cq, out.subbands[0], kplocs, 1,
                            output);
        
        // Read them out
        std::vector<float> descriptor = readBuffer<float>(cq, output);

        // Display
        for (float val: descriptor) {
            std::cout << val << std::endl;
        }
    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
                     
    return 0;
}


