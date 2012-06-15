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
        cl::Image2D inImage = createImage2D(context.context, bmp);

        std::cout << bmp.rows << " " << bmp.cols << std::endl;
        std::cout << "Creating Dtcwt" << std::endl;


        Dtcwt dtcwt(context.context, context.devices, cq);

        DtcwtTemps env = dtcwt.createContext(bmp.cols, bmp.rows,
                                           numLevels, startLevel);

        DtcwtOutput sbOutputs = {env};

        EnergyMap energyMap(context.context, context.devices);

        cl::Image2D emOut = createImage2D(context.context, bmp.cols / 2,
                                                   bmp.rows / 2);

        std::cout << "Running DTCWT" << std::endl;

        
        dtcwt(cq, inImage, env, sbOutputs);

        energyMap(cq, sbOutputs.subbands[0], emOut);

        cq.finish();

        std::cout << "Saving image" << std::endl;

        saveComplexImage("sb0.dat", cq, sbOutputs.subbands[0].sb[0]);
        saveComplexImage("sb1.dat", cq, sbOutputs.subbands[0].sb[1]);
        saveComplexImage("sb2.dat", cq, sbOutputs.subbands[0].sb[2]);
        saveComplexImage("sb3.dat", cq, sbOutputs.subbands[0].sb[3]);
        saveComplexImage("sb4.dat", cq, sbOutputs.subbands[0].sb[4]);
        saveComplexImage("sb5.dat", cq, sbOutputs.subbands[0].sb[5]);

        saveRealImage("em.dat", cq, emOut);

    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
                     
    return 0;
}



