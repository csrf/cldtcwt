#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "util/clUtil.h"
#include <iomanip>

#include <ctime>

#include "../BufferTrials/DTCWT/dtcwt.h"

#include <highgui.h>


int main()
{
    try {

        CLContext context;

        // Ready the command queue on the first device to hand
        cl::CommandQueue cq(context.context, context.devices[0]);

        const int numLevels = 6;
        const int startLevel = 0;


        //-----------------------------------------------------------------
        // Starting test code
  
        // Read in image
        cv::Mat bmp = cv::imread("test.bmp", 0);

        ImageBuffer<cl_float> inImage { 
            context.context, CL_MEM_READ_WRITE,
            bmp.cols, bmp.rows, 16, 32
        };

        // Upload the data
        cq.enqueueWriteBufferRect(inImage.buffer(), CL_TRUE,
              makeCLSizeT<3>({sizeof(float) * inImage.padding(),
                              inImage.padding(), 0}),
              makeCLSizeT<3>({0,0,0}),
              makeCLSizeT<3>({inImage.width() * sizeof(float),
                              inImage.height(), 1}),
              inImage.stride() * sizeof(float), 0,
              0, 0,
              bmp.ptr());

        std::cout << bmp.rows << " " << bmp.cols << std::endl;
        std::cout << "Creating Dtcwt" << std::endl;


        Dtcwt dtcwt(context.context, context.devices);

        DtcwtTemps env = dtcwt.createContext(bmp.cols, bmp.rows,
                                           numLevels, startLevel);

        DtcwtOutput sbOutputs = {env};


        std::cout << "Running DTCWT" << std::endl;

        
        dtcwt(cq, inImage, env, sbOutputs);


        cq.finish();

        std::cout << "Saving image" << std::endl;

        saveComplexImage("sb0.dat", cq, sbOutputs.subbands[0].sb[0]);
        saveComplexImage("sb1.dat", cq, sbOutputs.subbands[0].sb[1]);
        saveComplexImage("sb2.dat", cq, sbOutputs.subbands[0].sb[2]);
        saveComplexImage("sb3.dat", cq, sbOutputs.subbands[0].sb[3]);
        saveComplexImage("sb4.dat", cq, sbOutputs.subbands[0].sb[4]);
        saveComplexImage("sb5.dat", cq, sbOutputs.subbands[0].sb[5]);

    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
                     
    return 0;
}



