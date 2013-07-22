// Copyright (C) 2013 Timothy Gale
#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <tuple>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "util/clUtil.h"

#include "Filter/QuadToComplex/quadToComplex.h"

#include "Filter/referenceImplementation.h"

#include "Filter/imageBuffer.h"

// Check that the QuadToComplex kernel actually does what it should

std::tuple<Eigen::ArrayXXcf, Eigen::ArrayXXcf>
    quadToComplexGPU(const Eigen::ArrayXXf& in);


// Runs both with the same parameters, and displays output if failure,
// returning true.
bool compareImplementations(const Eigen::ArrayXXf& in, 
                            float tolerance);


int main()
{
    Eigen::ArrayXXf X1(8,16);
    X1.setRandom();

    float eps = 1.e-5;

    if (compareImplementations(X1, eps)) {
        std::cerr << "Failed quad-complex conversion"
                  << std::endl;
        return -1;
    }

    // No failures if we reached here
    return 0;
 
}



bool compareImplementations(const Eigen::ArrayXXf& in, 
                            float tolerance)
{
    Eigen::ArrayXXcf refSB0, refSB1,
                     gpuSB0, gpuSB1;

    // Try with reference and GPU implementations
    std::tie(refSB0, refSB1) = quadToComplex(in);
    std::tie(gpuSB0, gpuSB1) = quadToComplexGPU(in);
   
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
    quadToComplexGPU(const Eigen::ArrayXXf& in) 
{
    typedef
    Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Array;
        

    // Copy into an array where we set up the backing, so should
    // know the data format!
    std::vector<float> inValues(in.rows() * in.cols());
    Eigen::Map<Array> input(&inValues[0], in.rows(), in.cols());
    input = in;

    Eigen::ArrayXXcf sb0(in.rows() / 2, in.cols() / 2), 
                     sb1(in.rows() / 2, in.cols() / 2);

    const size_t outWidth = sb0.cols(),
                 outHeight = sb0.rows();

    // We need to read out the images (both real and imaginary)
    // then copy it over to sb0 or sb1.
    std::vector<Complex<cl_float>> outValues(outWidth * outHeight);

    try {

        CLContext context;

        // Ready the command queue on the first device to hand
        cl::CommandQueue cq(context.context, context.devices[0]);

        QuadToComplex quadToComplex(context.context, context.devices);

  
        const size_t width = in.cols(), height = in.rows(),
                     padding = 16, alignment = 32;

        ImageBuffer<cl_float> input(context.context, CL_MEM_READ_WRITE,
                                    width, height, padding, alignment); 


        ImageBuffer<Complex<cl_float>> 
                    sbImage(context.context, CL_MEM_READ_WRITE,
                            sb1.cols(), sb1.rows(), 0, 1, 2);

        // Upload the data
        input.write(cq, &inValues[0]);

        // Try the filter
        quadToComplex(cq, input, sbImage, 0, 1);

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




