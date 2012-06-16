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



        time_t start, end;
        const int numFrames = 1000;
        time(&start);
            for (int n = 0; n < numFrames; ++n) {
                dtcwt(cq, inImage, env, out);
                cq.finish();
            }
        time(&end);
        std::cout << (numFrames / difftime(end, start))
		  << " fps" << std::endl;
        std::cout << numFrames << " frames in " 
                  << difftime(end, start) << "s" << std::endl;

    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
                     
    return 0;
}


