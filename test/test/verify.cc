// Copyright (C) 2013 Timothy Gale
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "util/clUtil.h"
#include <iomanip>

#include <ctime>

#include "DTCWT/dtcwt.h"

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

        DtcwtTemps env {context.context, 
                        bmp.cols, bmp.rows,
                        startLevel, numLevels};

        DtcwtOutput sbOutputs = env.createOutputs();


        std::cout << "Running DTCWT" << std::endl;

        
        dtcwt(cq, inImage, env, sbOutputs);


        cq.finish();

        std::cout << "Saving image" << std::endl;

        ImageBuffer<Complex<cl_float>>
            sbSlice0 {sbOutputs[0], 0},
            sbSlice1 {sbOutputs[0], 1},
            sbSlice2 {sbOutputs[0], 2},
            sbSlice3 {sbOutputs[0], 3},
            sbSlice4 {sbOutputs[0], 4},
            sbSlice5 {sbOutputs[0], 5};
        saveComplexImage("sb0.dat", cq, sbSlice0);
        saveComplexImage("sb1.dat", cq, sbSlice1);
        saveComplexImage("sb2.dat", cq, sbSlice2);
        saveComplexImage("sb3.dat", cq, sbSlice3);
        saveComplexImage("sb4.dat", cq, sbSlice4);
        saveComplexImage("sb5.dat", cq, sbSlice5);

    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
                     
    return 0;
}



