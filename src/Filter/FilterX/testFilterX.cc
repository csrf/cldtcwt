#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <algorithm>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "util/clUtil.h"

#include "Filter/PadX/padX.h"
#include "Filter/FilterX/filterX.h"

#include "../referenceImplementation.h"



// Check that the FilterX kernel actually does what it should

Eigen::ArrayXXf convolveRowsGPU(const Eigen::ArrayXXf& in, 
                                const std::vector<float>& filter);


int main()
{

    std::vector<float> filter(13, 0.0);
    for (int n = 0; n < filter.size(); ++n)
        filter[n] = n + 1;

    Eigen::ArrayXXf X(5,12);
    X.setRandom();
    
    // Try with reference and GPU implementations
    Eigen::ArrayXXf refResult = convolveRows(X, filter);
    Eigen::ArrayXXf gpuResult = convolveRowsGPU(X, filter);
   
    // Check the maximum error is within tolerances
    float biggestDiscrepancy = 
        (refResult - gpuResult).abs().maxCoeff();

    // No problem if within tolerances
    if (biggestDiscrepancy < 1.e-5)
        return 0;
    else {

        // Display diagnostics:
        std::cerr << "Should have been:\n"
                  << refResult << "\n\n"
                  << "Was:\n"
                  << gpuResult << std::endl;

        return -1;
    }

}




Eigen::ArrayXXf convolveRowsGPU(const Eigen::ArrayXXf& in, 
                                const std::vector<float>& filter)
{
    typedef
    Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Array;
        

    // Copy into an array where we set up the backing, so should
    // know the data format!
    std::vector<float> inValues(in.rows() * in.cols());
    Eigen::Map<Array> input(&inValues[0], in.rows(), in.cols());
    input = in;

    std::vector<float> outValues(in.rows() * in.cols());
    Eigen::Map<Array> output(&outValues[0], in.rows(), in.cols());

    try {

        CLContext context;

        // Ready the command queue on the first device to hand
        cl::CommandQueue cq(context.context, context.devices[0]);


        PadX padX(context.context, context.devices);
        FilterX filterX(context.context, context.devices, filter);

  
        const size_t width = in.cols(), height = in.rows(),
                     padding = 16, alignment = 16;

        ImageBuffer<cl_float> input(context.context, CL_MEM_READ_WRITE,
                                    width, height, padding, alignment); 

        ImageBuffer<cl_float> output(context.context, CL_MEM_READ_WRITE,
                                     width, height, padding, alignment); 

        // Upload the data
        input.write(cq, &inValues[0]);

        // Try the filter
        padX(cq, input);
        filterX(cq, input, output);

        // Download the data
        output.read(cq, &outValues[0]);

    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
        throw;
    }

    Array out = output;

    return out;
}




