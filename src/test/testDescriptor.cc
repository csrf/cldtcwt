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

#include <cmath>
#include <stdexcept>

#include <highgui.h>
#include "KeypointDescriptor/extractDescriptors.h"


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

        DescriptorExtracter describer(context.context, context.devices,
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
                                         {56, 58});

        std::vector<cl_uint> kpOffsetsV
            = {0, 1};

        cl::Buffer kpOffsets = {context.context,
                                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                kpOffsetsV.size() * sizeof(cl_uint),
                                &kpOffsetsV[0]};

        cl::Buffer output = createBuffer(context.context, cq, 
                                 std::vector<float>(14 * 12));

        // Perform the transform
        dtcwt(cq, inImage, env, out);

        // Extract descriptors
        describer(cq, out.subbands[0], 4.0f,
                      out.subbands[1], 8.0f,
                      kplocs, 
                      kpOffsets, 0, 1,
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


