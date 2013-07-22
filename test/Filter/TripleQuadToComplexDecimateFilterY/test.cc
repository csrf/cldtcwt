// Copyright (C) 2013 Timothy Gale
#include <iostream>
#include <vector>
#include <array>
#include <stdexcept>
#include <algorithm>
#include <tuple>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "util/clUtil.h"

#include "Filter/TripleQuadToComplexDecimateFilterY/tripleQ2cDecimateFilterY.h"
#include "Filter/PadY/padY.h"

#include "Filter/referenceImplementation.h"

// Check that the QuadToComplex/DecimateFilterY combined kernel actually 
// does what it should

std::array<Eigen::ArrayXXcf, 6>
    tripleQuadToComplexDecimateFilterYGPU(const Eigen::ArrayXXf& in0,
                                    const Eigen::ArrayXXf& in1,
                                    const Eigen::ArrayXXf& in2,
                                    const std::vector<float>& filter0,
                                    bool swapOutputs0,
                                    const std::vector<float>& filter1,
                                    bool swapOutputs1,
                                    const std::vector<float>& filter2,
                                    bool swapOutputs2);


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
    Eigen::ArrayXXcf refSB0, refSB1;

    // Try with reference and GPU implementations
    Eigen::ArrayXXf filtered 
        = decimateConvolveCols(in, filter, swapOutputs);
    std::tie(refSB0, refSB1) = quadToComplex(filtered);

    std::array<Eigen::ArrayXXcf, 6> gpuSB
        = tripleQuadToComplexDecimateFilterYGPU(in, in, in, filter, swapOutputs,
                                              filter, swapOutputs,
                                              filter, swapOutputs);
   
    // Check the maximum error is within tolerances
    float biggestDiscrepancy = 
        std::max({ (refSB0 - gpuSB[0]).abs().maxCoeff(),
                   (refSB1 - gpuSB[5]).abs().maxCoeff(),
                   (refSB0 - gpuSB[1]).abs().maxCoeff(),
                   (refSB1 - gpuSB[4]).abs().maxCoeff(),
                   (refSB0 - gpuSB[2]).abs().maxCoeff(),
                   (refSB1 - gpuSB[3]).abs().maxCoeff() });

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
                  << "Was:\n";
        for (int n = 0; n < 6; ++n)
            std::cerr << gpuSB[n] << "\n\n";

        return true;
    }
}



std::array<Eigen::ArrayXXcf, 6>
    tripleQuadToComplexDecimateFilterYGPU(const Eigen::ArrayXXf& in0,
                                    const Eigen::ArrayXXf& in1,
                                    const Eigen::ArrayXXf& in2,
                                    const std::vector<float>& filter0,
                                    bool swapOutputs0,
                                    const std::vector<float>& filter1,
                                    bool swapOutputs1,
                                    const std::vector<float>& filter2,
                                    bool swapOutputs2)
{
    typedef
    Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Array;
        
    // Copy into an array where we set up the backing, so should
    // know the data format!
    std::vector<float> inValues(3 * in0.rows() * in0.cols());
    Eigen::Map<Array> input0(&inValues[0], in0.rows(), in0.cols()),
                      input1(&inValues[in0.rows() * in0.cols()], in0.rows(), in0.cols()),
                      input2(&inValues[2 * in0.rows() * in0.cols()], in0.rows(), in0.cols());

    input0 = in0;
    input1 = in1;
    input2 = in2;

    const size_t outWidth = in0.cols() / 2,
                 outHeight = ((in0.rows() % 4) == 2)?
                                (in0.rows() + 2) / 4
                              : in0.rows() / 4;

    std::array<Eigen::ArrayXXcf, 6> sb;
    for (auto& x: sb)
        x = Eigen::ArrayXXcf {outHeight, outWidth};

    // We need to read out the images (both real and imaginary)
    // then copy it over to sb0 or sb1.
    std::vector<Complex<cl_float>> outValues(outWidth * outHeight);

    try {

        CLContext context;

        // Ready the command queue on the first device to hand
        cl::CommandQueue cq(context.context, context.devices[0]);

        PadY padY(context.context, context.devices);
        TripleQuadToComplexDecimateFilterY 
            qtcDecFilterY(context.context, context.devices,
                          filter0, swapOutputs0,
                          filter1, swapOutputs1,
                          filter2, swapOutputs2);

  
        const size_t width = in0.cols(), height = in0.rows(),
                     padding = 16, alignment = 32;

        ImageBuffer<cl_float> input(context.context, CL_MEM_READ_WRITE,
                                    width, height, padding, alignment, 
                                    3); 

        // Add references to the individual slices
        ImageBuffer<cl_float> inputs[3]
            = {{input, 0}, {input, 1}, {input, 2}};

        ImageBuffer<Complex<cl_float>> sbImage(context.context, CL_MEM_READ_WRITE,
                                       sb[0].cols(), sb[0].rows(),
                                       0, alignment,
                                       6);

        // Upload the data
        input.write(cq, &inValues[0]);

        // Try the filter
        padY(cq, inputs[0]);
        padY(cq, inputs[1]);
        padY(cq, inputs[2]);
        qtcDecFilterY(cq, input, sbImage);

        // Download the data
        for (int n = 0; n < sbImage.numSlices(); ++n) {

            sbImage.read(cq, &outValues[0], {}, n);

            for (size_t r = 0; r < sb[n].rows(); ++r)
                for (size_t c = 0; c < sb[n].cols(); ++c) 
                    sb[n](r,c) = std::complex<float>
                        (outValues[r*sb[n].cols() + c].real,
                         outValues[r*sb[n].cols() + c].imag);

        }

    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
        throw;
    }

    return sb;
}




