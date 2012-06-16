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
        const int startLevel = 0;

        // Read the image in
        cv::Mat bmp = cv::imread(filename, 0) / 255.0f;
        cl::Image2D inImage = createImage2D(context.context, bmp);

        // Create the DTCWT itself
        Dtcwt dtcwt(context.context, context.devices, cq);

        // Create the intermediate storage for the DTCWT
        DtcwtTemps env = dtcwt.createContext(bmp.cols, bmp.rows,
                                             numLevels, startLevel);

        // Create the outputs storage for the DTCWT
        DtcwtOutput sbOutputs = {env};

        // Perform DTCWT
        dtcwt(cq, inImage, env, sbOutputs);
        cq.finish();

        // Produce the outputs
        for (int l = 0; l < numLevels; ++l) {
            for (int sb = 0; sb < 6; ++sb) {

                // Construct output name in format 
                // <original name>.<level>.<subband number>
                std::ostringstream ss;
                ss << filename << "." << l << "." << sb;

                saveComplexImage(ss.str(), 
                                 cq, 
                                 sbOutputs.subbands[l].sb[sb]);
            }
        }

        saveRealImage("in.dat", cq, inImage);
    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
                     
    return 0;
}


