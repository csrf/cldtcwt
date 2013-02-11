#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "util/clUtil.h"


#include "BufferTrials/PadY/padY.h"
#include "BufferTrials/FilterY/filterY.h"

#include "../referenceImplementation.h"


// Check that the FilterY kernel actually does what it should

Eigen::ArrayXXf convolveColsGPU(const Eigen::ArrayXXf& in, 
                                const std::vector<float>& filter);


int main()
{

    std::vector<float> filter(13, 0.0);
    for (int n = 0; n < filter.size(); ++n)
        filter[n] = n + 1;

    Eigen::ArrayXXf X(18,5);
    X.setRandom();
    
    // Try with reference and GPU implementations
    Eigen::ArrayXXf refResult = convolveCols(X, filter);
    Eigen::ArrayXXf gpuResult = convolveColsGPU(X, filter);
   
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






Eigen::ArrayXXf convolveColsGPU(const Eigen::ArrayXXf& in, 
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

        PadY padY(context.context, context.devices);
        FilterY filterY(context.context, context.devices, filter);

  
        const size_t width = in.cols(), height = in.rows(),
                     padding = 16, alignment = 16;

        ImageBuffer<cl_float> input(context.context, CL_MEM_READ_WRITE,
                                    width, height, padding, alignment); 

        ImageBuffer<cl_float> output(context.context, CL_MEM_READ_WRITE,
                                     width, height, padding, alignment); 

        // Upload the data
        cq.enqueueWriteBufferRect(input.buffer(), CL_TRUE,
              makeCLSizeT<3>({sizeof(float) * input.padding(),
                              input.padding(), 0}),
              makeCLSizeT<3>({0,0,0}),
              makeCLSizeT<3>({input.width() * sizeof(float),
                              input.height(), 1}),
              input.stride() * sizeof(float), 0,
              0, 0,
              &inValues[0]);

        // Try the filter
        padY(cq, input);
        filterY(cq, input, output);

        // Download the data
        cq.enqueueReadBufferRect(output.buffer(), CL_TRUE,
              makeCLSizeT<3>({sizeof(float) * output.padding(),
                             output.padding(), 0}),
              makeCLSizeT<3>({0,0,0}),
              makeCLSizeT<3>({output.width() * sizeof(float),
                             output.height(), 1}),
              output.stride() * sizeof(float), 0,
              0, 0,
              &outValues[0]);

    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
        throw;
    }

    Array out = output;

    return out;
}




