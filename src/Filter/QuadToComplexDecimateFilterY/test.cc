#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <tuple>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "util/clUtil.h"

#include "q2cDecimateFilterY.h"
#include "../PadY/padY.h"

#include "../referenceImplementation.h"

// Check that the QuadToComplex/DecimateFilterY combined kernel actually 
// does what it should

std::tuple<Eigen::ArrayXXcf, Eigen::ArrayXXcf>
    quadToComplexDecimateFilterYGPU(const Eigen::ArrayXXf& in,
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


    Eigen::ArrayXXf X1(16,22);
    X1.setRandom();

    float eps = 1.e-3;

    if (compareImplementations(X1, filter, false, eps)) {
        std::cerr << "Failed decimation and quad-complex"
                  << std::endl;
        return -1;
    }

    if (compareImplementations(X1, filter, true, eps)) {
        std::cerr << "Failed decimation and quad-complex, swapped "
                     "trees at output"
                  << std::endl;
        return -1;
    }

    Eigen::ArrayXXf X2(6,24);
    X2.setRandom();

    if (compareImplementations(X2, filter, false, eps)) {
        std::cerr << "Failed decimation and quad-complex "
                     "with symmetric extension"
                  << std::endl;
        return -1;
    }

    if (compareImplementations(X2, filter, true, eps)) {
        std::cerr << "Failed decimation and quad-complex "
                     "with symmetric extension and trees swapped "
                     "at output"
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
    Eigen::ArrayXXcf refSB0, refSB1,
                     gpuSB0, gpuSB1;

    // Try with reference and GPU implementations
    Eigen::ArrayXXf filtered 
        = decimateConvolveCols(in, filter, swapOutputs);
    std::tie(refSB0, refSB1) = quadToComplex(filtered);

    std::tie(gpuSB0, gpuSB1)
        = quadToComplexDecimateFilterYGPU(in, filter, swapOutputs);
   
    // Check the maximum error is within tolerances
    float biggestDiscrepancy = 
        std::max((refSB0 - gpuSB0).abs().maxCoeff(),
                 (refSB1 - gpuSB1).abs().maxCoeff());

    // No problem if within tolerances
    if (biggestDiscrepancy < tolerance)
        return false;
    else {

        // Display diagnostics:
        std::cerr << "Input:\n"
                  << in << "\n\n"
                  << "Should have been:\n"
                  << refSB0 << "\n\n"
                  << refSB1 << "\n\n"
                  << "Was:\n"
                  << gpuSB0 << "\n\n"
                  << gpuSB1 << "\n\n"
                  << std::endl;

        return true;
    }
}



std::tuple<Eigen::ArrayXXcf, Eigen::ArrayXXcf>
    quadToComplexDecimateFilterYGPU(const Eigen::ArrayXXf& in,
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

    const size_t outWidth = in.cols() / 2,
                 outHeight = ((in.rows() % 4) == 2)?
                                (in.rows() + 2) / 4
                              : in.rows() / 4;

    Eigen::ArrayXXcf sb0(outHeight, outWidth), 
                     sb1(outHeight, outWidth);


    // We need to read out the images (both real and imaginary)
    // then copy it over to sb0 or sb1.
    std::vector<Complex<cl_float>> outValues(outWidth * outHeight);

    try {

        CLContext context;

        // Ready the command queue on the first device to hand
        cl::CommandQueue cq(context.context, context.devices[0]);

        PadY padY(context.context, context.devices);
        QuadToComplexDecimateFilterY 
            qtcDecFilterY(context.context, context.devices,
                          filter, swapOutputs);

  
        const size_t width = in.cols(), height = in.rows(),
                     padding = 16, alignment = 32;

        ImageBuffer<cl_float> input(context.context, CL_MEM_READ_WRITE,
                                    width, height, padding, alignment); 


        ImageBuffer<Complex<cl_float>> sbImage(context.context, CL_MEM_READ_WRITE,
                                       sb0.cols(), sb0.rows(),
                                       0, alignment,
                                       2);

        // Upload the data
        input.write(cq, &inValues[0]);

        // Try the filter
        padY(cq, input);
        qtcDecFilterY(cq, input, sbImage, 0, 1);

        // Download the data
        sbImage.read(cq, &outValues[0], {}, 0);

        for (size_t r = 0; r < sb0.rows(); ++r)
            for (size_t c = 0; c < sb0.cols(); ++c) 
                sb0(r,c) = std::complex<float>
                    (outValues[r*sb0.cols() + c].real,
                     outValues[r*sb0.cols() + c].imag);

        sbImage.read(cq, &outValues[0], {}, 1);

        for (size_t r = 0; r < sb1.rows(); ++r)
            for (size_t c = 0; c < sb1.cols(); ++c) 
                sb1(r,c) = std::complex<float>
                    (outValues[r*sb1.cols() + c].real,
                     outValues[r*sb1.cols() + c].imag);

    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
        throw;
    }

    return std::make_tuple(sb0, sb1);
}




