// Copyright (C) 2013 Timothy Gale
#include <iostream>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "util/clUtil.h"

#include <chrono>
typedef std::chrono::duration<double>
    DurationSeconds;

#include "Filter/QuadToComplex/quadToComplex.h"


#include <sstream>

template <typename T>
T readStr(const char* string)
{
    std::istringstream s(string);

    T result;
    s >> result;
    return result;
}


int main(int argc, const char* argv[])
{
    // Measure the speed of the quad to complex operation

    size_t width = 1280 / 2, height = 720, numIterations = 1000;

    // First and second arguments: width and height
    if (argc > 2) {
        width = readStr<size_t>(argv[1]);
        height = readStr<size_t>(argv[2]);
    }

    // Fourth argument: number of iterations
    if (argc > 3) {
        numIterations = readStr<size_t>(argv[3]);
    }


    try {

        CLContext context;

        // Ready the command queue on the first device to hand
        cl::CommandQueue cq(context.context, context.devices[0]);

        QuadToComplex qtc {context.context, context.devices};
  
        const size_t padding = 16, alignment = 2*16;

        // Create input and output buffers
        ImageBuffer<cl_float> 
            input {context.context, CL_MEM_READ_WRITE,
                   width, height, padding, alignment};

        ImageBuffer<Complex<cl_float>> 
            sb {context.context, CL_MEM_READ_WRITE, 
                width / 2, height / 2, 0, 1,
                2};

        {
            // Run, timing
            auto start = std::chrono::system_clock::now();

            for (int n = 0; n < numIterations; ++n) 
                qtc(cq, input, sb, 0, 1);

            cq.finish();
            auto end = std::chrono::system_clock::now();

            // Work out what the difference between these is
            double t = DurationSeconds(end - start).count();

            std::cout << "QuadToComplex: " 
                    << (t / numIterations * 1000) << " ms" << std::endl;
        }
    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
                     
    return 0;
}


