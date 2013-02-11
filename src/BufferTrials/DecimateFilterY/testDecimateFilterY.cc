#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "util/clUtil.h"


#include "BufferTrials/PadY/padY.h"
#include "BufferTrials/DecimateFilterY/decimateFilterY.h"

#include "../referenceImplementation.h"

// Check that the DecimateFilterY kernel actually does what it should

Eigen::ArrayXXf decimateConvolveColsGPU(const Eigen::ArrayXXf& in, 
                                const std::vector<float>& filter,
                                bool swapOutputs);


// Runs both with the same parameters, and displays output if failure,
// returning true.
bool compareImplementations(const Eigen::ArrayXXf& in, 
                            const std::vector<float>& filter,
                            bool swapOutputs,
                            float tolerance);


int main()
{

    std::vector<float> filter(14, 0.0);
    for (int n = 0; n < filter.size(); ++n)
        filter[n] = n + 1;

    
    Eigen::ArrayXXf X1(16, 5);
    X1.setRandom();

    float eps = 1.e-5;

    if (compareImplementations(X1, filter, false, eps)) {
        std::cerr << "Failed no extension, no swapped outputs" 
                  << std::endl;
        return -1;
    }

    if (compareImplementations(X1, filter, true, eps)) {
        std::cerr << "Failed no extension, swapped output trees" 
                  << std::endl;
        return -1;
    }
    
    Eigen::ArrayXXf X2(18, 5);
    X2.setRandom();

    if (compareImplementations(X2, filter, false, eps)) {
        std::cerr << "Failed extension, no swapped outputs" 
                  << std::endl;
        return -1;
    }

    if (compareImplementations(X2, filter, true, eps)) {
        std::cerr << "Failed extension, swapped output trees" 
                  << std::endl;
        return -1;
    }

    // No failures if we reached here
    return 0;
 
}



bool compareImplementations(const Eigen::ArrayXXf& in, 
                            const std::vector<float>& filter,
                            bool swapOutputs,
                            float tolerance)
{
    // Try with reference and GPU implementations
    Eigen::ArrayXXf refResult = decimateConvolveCols(in, filter, swapOutputs);
    Eigen::ArrayXXf gpuResult = decimateConvolveColsGPU(in, filter, swapOutputs);
   
    // Check the maximum error is within tolerances
    float biggestDiscrepancy = 
        (refResult - gpuResult).abs().maxCoeff();

    // No problem if within tolerances
    if (biggestDiscrepancy < tolerance)
        return false;
    else {

        // Display diagnostics:
        std::cerr << "Input:\n"
                  << in << "\n\n"
                  << "Should have been:\n"
                  << refResult << "\n\n"
                  << "Was:\n"
                  << gpuResult << std::endl;

        return true;
    }
}






Eigen::ArrayXXf decimateConvolveColsGPU(const Eigen::ArrayXXf& in, 
                                const std::vector<float>& filter,
                                bool swapOutputs)
{
    typedef
    Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Array;
        

    // Copy into an array where we set up the backing, so should
    // know the data format!
    std::vector<float> inValues(in.rows() * in.cols());
    Eigen::Map<Array> input(&inValues[0], in.rows(), in.cols());
    input = in;


    const size_t outputHeight = (in.rows() + in.rows() % 4) / 2;
    std::vector<float> outValues(outputHeight * in.cols());
    Eigen::Map<Array> output(&outValues[0], outputHeight, in.cols());

    try {

        CLContext context;

        // Ready the command queue on the first device to hand
        cl::CommandQueue cq(context.context, context.devices[0]);

        PadY padY(context.context, context.devices);
        DecimateFilterY decimateFilterY(context.context, context.devices, filter,
                                        swapOutputs);

  
        const size_t width = in.cols(), height = in.rows(),
                     padding = 16, alignment = 32;

        ImageBuffer<cl_float> input(context.context, CL_MEM_READ_WRITE,
                                    width, height, padding, alignment); 

        ImageBuffer<cl_float> output(context.context, CL_MEM_READ_WRITE,
                                     width, outputHeight, padding, alignment); 

        // Upload the data
        input.write(cq, &inValues[0]);

        // Try the filter
        padY(cq, input);
        decimateFilterY(cq, input, output);

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




