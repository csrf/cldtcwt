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

#include <sstream>

template <typename T>
T readStr(const char* string)
{
    std::istringstream s(string);

    T result;
    s >> result;
    return result;
}


int main(int argc, const char* argv[])
{
    // Measure the speed of the DTCWT, defaulting to these parameters:
    size_t width = 1280, height = 720, numLevels = 6, numIterations = 1000;

    // First and second arguments: width and height
    if (argc > 2) {
        width = readStr<size_t>(argv[1]);
        height = readStr<size_t>(argv[2]);
    }

    // Third argument: number of levels to calculate
    if (argc > 3) {
        numLevels = readStr<size_t>(argv[3]);
    }

    // Fourth argument: number of iterations
    if (argc > 4) {
        numIterations = readStr<size_t>(argv[4]);
    }


    try {

        CLContext context;

        // Ready the command queue on the first device to hand
        cl::CommandQueue cq(context.context, context.devices[0]);


        const int startLevel = 1;


        //-----------------------------------------------------------------
        // Starting test code
  
        // Read in image
        cv::Mat bmp = cv::imread("testDTCWT.bmp", 0);
        ImageBuffer<cl_float> inImage { 
            context.context, CL_MEM_READ_WRITE,
            width, height, 16, 32
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

        std::cout << "Creating Dtcwt" << std::endl;

        Dtcwt dtcwt(context.context, context.devices);

        std::cout << "Creating the DTCWT environment..." << std::endl;

        DtcwtTemps env = dtcwt.createContext(inImage.width(), inImage.height(),
                                           numLevels, startLevel);

        std::cout << "Creating the subband output images..." << std::endl;
        DtcwtOutput out(env);

        std::cout << "Running DTCWT" << std::endl;

        auto start = std::chrono::steady_clock::now();

        dtcwt(cq, inImage, env, out);
        for (int n = 0; n < (numIterations-1); ++n) 
            dtcwt(cq, inImage, env, out); //, out.subbands.back().done);
        cq.finish();

        auto end = std::chrono::steady_clock::now();

        // Work out what the difference between these is
        double t = DurationMilliseconds(end - start).count();

        std::cout << (numIterations / (t / 1000.f))
		  << " fps" << std::endl;
        std::cout << (t / numIterations) << "ms per iteration"
                  << std::endl;

    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
                     
    return 0;
}


