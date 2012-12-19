#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <algorithm>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "util/clUtil.h"

#include <sys/timeb.h>

#include "BufferTrials/FilterX/filterX.h"

#include <Eigen/Dense>

// Check that the FilterX kernel actually does what it should

Eigen::ArrayXXf convolveRows(const Eigen::ArrayXXf& in, 
                             const std::vector<float>& filter);

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



Eigen::ArrayXXf convolveRows(const Eigen::ArrayXXf& in, 
                             const std::vector<float>& filter)
{
    size_t offset = (filter.size() - 1) / 2;

    Eigen::ArrayXXf output(in.rows(), in.cols());

    // Pad the input
    Eigen::ArrayXXf padded(in.rows(), in.cols() + filter.size() - 1);

    for (int n = 0; n < padded.cols(); ++n) 
        padded.col(n) = in.col(wrap(n - offset, in.cols()));

    // For each output pixel
    for (size_t r = 0; r < in.rows(); ++r)
        for (size_t c = 0; c < in.cols(); ++c) {

            // Perform the convolution
            float v = 0.f;
            for (size_t n = 0; n < filter.size(); ++n)
                v += filter[filter.size()-n-1]
                        * padded(r, c+n);

            output(r,c) = v;
        }

    return output;
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

        FilterX filterX(context.context, context.devices, filter);

  
        const size_t width = in.cols(), height = in.rows(),
                     padding = 8, alignment = 8;

        ImageBuffer input(context.context, CL_MEM_READ_WRITE,
                          width, height, padding, alignment); 

        ImageBuffer output(context.context, CL_MEM_READ_WRITE,
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
        filterX(cq, input, output);

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




