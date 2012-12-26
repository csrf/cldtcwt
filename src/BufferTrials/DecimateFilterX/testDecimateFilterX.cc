#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <algorithm>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "util/clUtil.h"

#include <sys/timeb.h>

#include "BufferTrials/PadX/padX.h"
#include "BufferTrials/DecimateFilterX/decimateFilterX.h"

#include <Eigen/Dense>

// Check that the FilterX kernel actually does what it should

Eigen::ArrayXXf decimateConvolveRows(const Eigen::ArrayXXf& in, 
                             const std::vector<float>& filter,
                             bool swapOutputs);

Eigen::ArrayXXf decimateConvolveRowsGPU(const Eigen::ArrayXXf& in, 
                                const std::vector<float>& filter,
                                bool swapOutputs);


int main()
{

    std::vector<float> filter(14, 0.0);
    for (int n = 0; n < filter.size(); ++n)
        filter[n] = n + 1;

    Eigen::ArrayXXf X(5,16);
    X.setRandom();
    
    // Try with reference and GPU implementations
    Eigen::ArrayXXf refResult = decimateConvolveRows(X, filter, false);
    Eigen::ArrayXXf gpuResult = decimateConvolveRowsGPU(X, filter, false);
   
    // Check the maximum error is within tolerances
    float biggestDiscrepancy = 
        (refResult - gpuResult).abs().maxCoeff();

    // No problem if within tolerances
    if (biggestDiscrepancy < 1.e-5)
        return 0;
    else {

        // Display diagnostics:
        std::cerr << "Input:\n"
                  << X << "\n\n"
                  << "Should have been:\n"
                  << refResult << "\n\n"
                  << "Was:\n"
                  << gpuResult << std::endl;

        return -1;
    }

}



unsigned int wrap(int n, int width)
{
    // Wrap so that the pattern goes
    // forwards-backwards-forwards-backwards etc, with the end
    // values repeated.
    
    int result = n % (2 * width);

    // Make sure we get the positive result
    if (result < 0)
        result += 2*width;

    return std::min(result, 2*width - result - 1);
}



Eigen::ArrayXXf decimateConvolveRows
                            (const Eigen::ArrayXXf& in, 
                             const std::vector<float>& filter,
                             bool swapOutputs)
{
    // If extending, we want to create an extra output by
    // taking an extra sample from each end.  Symmetric is
    // whether the reversed filter output should come first in
    // the pairs or second

    bool extend = (in.cols() % 4) != 0;

    size_t offset = filter.size() - 2 + (extend? 1 : 0);

    Eigen::ArrayXXf output(in.rows(), 
            (in.cols() + (extend? 2 : 0)) / 2);

    // Pad the input
    Eigen::ArrayXXf padded(in.rows(), in.cols() + 2 * offset);

    for (int n = 0; n < padded.cols(); ++n) 
        padded.col(n) = in.col(wrap(n - offset, in.cols()));

    // For each pair of output pixels
    for (size_t r = 0; r < output.rows(); ++r)
        for (size_t c = 0; c < output.cols(); c += 2) {

            // Perform the convolution
            float v1 = 0.f, v2 = 0.f;

            for (size_t n = 0; n < filter.size(); ++n) {
                v1 += filter[filter.size()-n-1]
                        * padded(r, 2*c+2*n);
               
                v2 += filter[n]
                        * padded(r, 2*c+2*n+1);
            }

            output(r,c) = swapOutputs? v2 : v1;
            output(r,c+1) = swapOutputs? v1 : v2;
        }

    return output;
}



Eigen::ArrayXXf decimateConvolveRowsGPU(const Eigen::ArrayXXf& in, 
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


    const size_t outputWidth = (in.cols() + in.cols() % 4) / 2;
    std::vector<float> outValues(in.rows() * outputWidth);
    Eigen::Map<Array> output(&outValues[0], in.rows(), outputWidth);

    try {

        CLContext context;

        // Ready the command queue on the first device to hand
        cl::CommandQueue cq(context.context, context.devices[0]);

        PadX padX(context.context, context.devices);
        DecimateFilterX decimateFilterX(context.context, context.devices, filter,
                                        swapOutputs);

  
        const size_t width = in.cols(), height = in.rows(),
                     padding = 16, alignment = 32;

        ImageBuffer input(context.context, CL_MEM_READ_WRITE,
                          width, height, padding, alignment); 

        ImageBuffer output(context.context, CL_MEM_READ_WRITE,
                           outputWidth, height, padding, alignment); 

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
        padX(cq, input);
        decimateFilterX(cq, input, output);

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




