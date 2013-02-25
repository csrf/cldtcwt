#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "util/clUtil.h"
#include "DTCWT/dtcwt.h"
#include <iomanip>

#include <chrono>
typedef std::chrono::duration<double, std::milli>
    DurationMilliseconds;

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
        ImageBuffer<cl_float> inImage { 
            context.context, CL_MEM_READ_WRITE,
            /*bmp.cols*/1280, /*bmp.rows*/720, 16, 32
        };

        // Upload the data
/*        cq.enqueueWriteBufferRect(inImage.buffer(), CL_TRUE,
              makeCLSizeT<3>({sizeof(float) * inImage.padding(),
                              inImage.padding(), 0}),
              makeCLSizeT<3>({0,0,0}),
              makeCLSizeT<3>({inImage.width() * sizeof(float),
                              inImage.height(), 1}),
              inImage.stride() * sizeof(float), 0,
              0, 0,
              bmp.ptr());*/

        std::cout << bmp.rows << " " << bmp.cols << std::endl;
        std::cout << "Creating Dtcwt" << std::endl;

        Dtcwt dtcwt(context.context, context.devices);

        std::cout << "Creating the DTCWT environment..." << std::endl;

        DtcwtTemps env = dtcwt.createContext(inImage.width(), inImage.height(),
                                           numLevels, startLevel);

        std::cout << "Creating the subband output images..." << std::endl;
        DtcwtOutput out(env);

        std::cout << "Running DTCWT" << std::endl;

        const int numFrames = 100;
        auto start = std::chrono::steady_clock::now();

        dtcwt(cq, inImage, env, out);
        for (int n = 0; n < (numFrames-1); ++n) 
            dtcwt(cq, inImage, env, out); //, out.subbands.back().done);
        cq.finish();

        auto end = std::chrono::steady_clock::now();

        // Work out what the difference between these is
        double t = DurationMilliseconds(end - start).count();

        std::cout << (numFrames / (t / 1000.f))
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


