#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

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
    // Extracts descriptors from an image
    //
    // Arguments: image filename
    // list of coordinates, x y x y x y etc.
    
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
        Dtcwt dtcwt(context.context, context.devices);
        DescriptorExtracter describer(context.context, context.devices, 2);


        // Read the image in
        cv::Mat bmp = cv::imread(argv[1], 0);
        cv::Mat floatBmp;
        bmp.convertTo(floatBmp, CV_32F);


        // Upload said image
        ImageBuffer<cl_float> inImage(context.context, CL_MEM_READ_WRITE,
                                      bmp.cols, bmp.rows,
                                      16, 32);
        inImage.write(cq, reinterpret_cast<cl_float*>(floatBmp.ptr()));


        // Parse sampling locations
        std::vector<cl_float> kplocsV;
        const size_t numKeypoints = (argc-2) / 2;
        for (int n = 0; n < (2*numKeypoints); ++n) {
            cl_float val;
            std::istringstream ss(argv[n+2]);
            ss >> val;
            
            kplocsV.push_back(val);
        }

        
        // Upload sampling locations
        cl::Buffer kplocs = createBuffer(context.context, cq, kplocsV);
        std::vector<cl_uint> kpOffsetsV = {0, numKeypoints};
        cl::Buffer kpOffsets = {context.context,
                                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                kpOffsetsV.size() * sizeof(cl_uint),
                                &kpOffsetsV[0]};


        // Create temporaries and outputs for the DTCWT
        DtcwtTemps env = dtcwt.createContext(bmp.cols, bmp.rows,
                                           numLevels, startLevel);
        DtcwtOutput out(env);

        
        // Create output buffer for the descriptors
        cl::Buffer output = createBuffer(context.context, cq, 
                             std::vector<float>(14 * 12 * numKeypoints));


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


