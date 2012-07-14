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

#include <cmath>
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

        const float pi = 4 * atan(1);

        // Pattern of locations to sample at
        // Set up the centre
        std::vector<Coord> pattern = {{0 ,0}};

        // Set up the circle
        for (int n = 0; n < 12; ++n) {
            pattern.push_back({float(sin(float(n) / 12.f * 2.f * pi)),
                               float(cos(float(n) / 12.f * 2.f * pi))});
        }


        // Ready the keypoint extractor
        DescriptorExtracter
            descriptorExtracter(context.context, context.devices, cq,
                                pattern, 1.f,
                                13, 0,
                                2);

        // Read in image
        cv::Mat bmp = cv::imread(argv[1], 0) / 255.0f;
        cl::Image2D inImage = createImage2D(context.context, bmp);

        // Create temporaries and outputs for the DTCWT
        DtcwtTemps env = dtcwt.createContext(bmp.cols, bmp.rows,
                                           numLevels, startLevel);
        DtcwtOutput out(env);

        // Create locations to sample at
        cl::Buffer kplocs = createBuffer(context.context, cq, 
                                         {14.f, 14.5f});

        cl::Buffer output = createBuffer(context.context, cq, 
                                 std::vector<float>(pattern.size() * 12));

        // Perform the transform
        dtcwt(cq, inImage, env, out);

        // Extract descriptors
        descriptorExtracter(cq, out.subbands[0], kplocs, 2,
                            output);
        
        // Read them out
        saveComplexBuffer("interpolations.dat", cq, output);

    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
                     
    return 0;
}


