#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "DTCWT/dtcwt.h"


#include "util/clUtil.h"
#include <iomanip>

#include <ctime>

#include <stdexcept>

#include <highgui.h>
#include <sstream>


int main(int argc, char** argv)
{
    // Make sure we were given the image name to process
    if (argc < 2) {
        std::cerr << "No image provided!" << std::endl;
        return -1;
    }

    std::string filename(argv[1]);

    try {

        CLContext context;

        // Ready the command queue on the first device to hand
        cl::CommandQueue cq(context.context, context.devices[0]);

        // This should make sure all code gets exercised, including the
        // lolo decimated
        const int numLevels = 3;
        const int startLevel = 1;

        // Read the image in
        cv::Mat bmp = cv::imread(filename, 0);
        cv::Mat floatBmp;
        bmp.convertTo(floatBmp, CV_32F);
        floatBmp /= 255.f;


        ImageBuffer<cl_float> inImage { 
            context.context, CL_MEM_READ_WRITE,
            floatBmp.cols, floatBmp.rows, 16, 32
        };

        inImage.write(cq, reinterpret_cast<cl_float*>(floatBmp.ptr()));


        // Create the DTCWT itself
        Dtcwt dtcwt(context.context, context.devices);

        // Create the intermediate storage for the DTCWT
        DtcwtTemps env {context.context,
                        floatBmp.cols, floatBmp.rows,
                        startLevel, numLevels};

        // Create the outputs storage for the DTCWT
        DtcwtOutput sbOutputs = env.createOutputs();

        // Perform DTCWT
        dtcwt(cq, inImage, env, sbOutputs);
        cq.finish();

        // Produce the outputs
        for (int l = 0; l < numLevels; ++l) {
            for (int sb = 0; sb < 6; ++sb) {

                // Construct output name in format 
                // <original name>.<level>.<subband number>
                std::ostringstream ss;
                ss << filename << "." << (l + sbOutputs.startLevel()) << "." << sb;

                saveComplexImage(ss.str(), 
                                 cq, 
                                 sbOutputs[l][sb]);
            }
        }

        //saveRealImage("in.dat", cq, inImage);
        //saveRealImage("lo.dat", cq, env.levelTemps[0].lo);
    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
                     
    return 0;
}


