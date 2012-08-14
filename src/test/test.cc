#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "DTCWT/filterer.h"
#include "util/clUtil.h"
#include "DTCWT/dtcwt.h"
#include <iomanip>

#include <sys/timeb.h>

#include <stdexcept>

#include <highgui.h>
#include "KeypointDescriptor/extractDescriptors.h"


int main()
{
    try {

        CLContext context;

        // Ready the command queue on the first device to hand
        cl::CommandQueue cq(context.context, context.devices[0]);


        const int numLevels = 6;
        const int startLevel = 1;


        //-----------------------------------------------------------------
        // Starting test code
  
        // Read in image
        cv::Mat bmp = cv::imread("testDTCWT.bmp", 0);
        cl::Image2D inImage = createImage2D(context.context, bmp);

        std::cout << bmp.rows << " " << bmp.cols << std::endl;
        std::cout << "Creating Dtcwt" << std::endl;

        Dtcwt dtcwt(context.context, context.devices, cq);

        std::cout << "Creating the DTCWT environment..." << std::endl;

        DtcwtTemps env = dtcwt.createContext(bmp.cols, bmp.rows,
                                           numLevels, startLevel);

        std::cout << "Creating the subband output images..." << std::endl;
        DtcwtOutput out(env);

        std::cout << "Running DTCWT" << std::endl;



        timeb start, end;
        const int numFrames = 1000;
        ftime(&start);
            for (int n = 0; n < numFrames; ++n) {
                dtcwt(cq, inImage, env, out);
                cq.finish();
            }
        ftime(&end);

        // Work out what the difference between these is
        double t = end.time - start.time 
                 + 0.001 * (end.millitm - start.millitm);

        std::cout << (numFrames / t)
		  << " fps" << std::endl;
        std::cout << numFrames << " frames in " 
                  << t << "s" << std::endl;

    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
                     
    return 0;
}


