#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "util/clUtil.h"
#include <iomanip>


#include "MiscKernels/accumulate.h"



int main()
{

    CLContext context;

    // Ready the command queue on the first device to hand
    cl::CommandQueue cq(context.context, context.devices[0]);

    //-----------------------------------------------------------------
    // Starting test code
    
    Accumulate accumulate(context.context, context.devices);

    // Test input
    std::vector<cl_uint> counts = {12, 5, 1, 3};

    // Create inputs and outputs
    cl::Buffer countsInput = {
        context.context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,      // Flags
        counts.size() * sizeof(cl_uint), // Size to allocate
        &counts[0]
    };

    cl::Buffer cumSumOutput = {
        context.context,
        CL_MEM_READ_WRITE,              // Flags
        (counts.size() + 1) * sizeof(cl_uint), // Size to allocate
        nullptr
    };

    // Execute
    accumulate(cq, countsInput, cumSumOutput, 20);

    // Read out
    std::vector<cl_uint> cumSums(counts.size() + 1);
    cq.enqueueReadBuffer(cumSumOutput, CL_TRUE, 0, 
                         cumSums.size() * sizeof(cl_uint), &cumSums[0]);

    cq.finish();

    for (cl_uint n: cumSums)
        std::cout << n << std::endl;

                     
    return 0;
}


