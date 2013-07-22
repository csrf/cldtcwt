// Copyright (C) 2013 Timothy Gale
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "util/clUtil.h"
#include <iomanip>


#include "KeypointDetector/Concat/concat.h"



int main()
{

    CLContext context;

    // Ready the command queue on the first device to hand
    cl::CommandQueue cq(context.context, context.devices[0]);

    //-----------------------------------------------------------------
    // Starting test code
    
    Concat concat(context.context, context.devices);

    // Test input
    std::vector<float> valuesV = {1, 2, 3, 4, 5, 6, 7, 8};

    // Create inputs and outputs
    cl::Buffer values = {
        context.context,
        CL_MEM_READ_WRITE,      // Flags
        valuesV.size() * sizeof(float), // Size to allocate
    };

    cq.enqueueWriteBuffer(values, CL_TRUE, 
                          0, values.getInfo<CL_MEM_SIZE>(), 
                          &valuesV[0]);

    std::vector<cl_uint> cumCountV = {0, 4, 5};
    cl::Buffer cumCount = {
        context.context,
        CL_MEM_READ_WRITE,              // Flags
        cumCountV.size() * sizeof(cl_uint) // Size to allocate
    };

    cq.enqueueWriteBuffer(cumCount, CL_TRUE, 
                          0, cumCount.getInfo<CL_MEM_SIZE>(), 
                          &cumCountV[0]);

    std::vector<float> outputV(2 * 5);
    cl::Buffer output = {
        context.context,
        CL_MEM_READ_WRITE,              // Flags
        outputV.size() * sizeof(float) // Size to allocate
    };

    cq.finish();

    // Execute
    concat(cq, values, output, cumCount, 0, 2);
    concat(cq, values, output, cumCount, 1, 2);

    cq.finish();

    // Read out
    cq.enqueueReadBuffer(output, CL_TRUE, 0, 
                         output.getInfo<CL_MEM_SIZE>(), &outputV[0]);

    for (cl_uint n: outputV)
        std::cout << n << " ";
    std::cout << std::endl;

                     
    return 0;
}


