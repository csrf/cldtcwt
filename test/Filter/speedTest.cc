// Copyright (C) 2013 Timothy Gale
#include <iostream>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "util/clUtil.h"

#include <chrono>
typedef std::chrono::duration<double>
    DurationSeconds;

#include "Filter/PadX/padX.h"
#include "Filter/PadY/padY.h"
#include "Filter/FilterX/filterX.h"
#include "Filter/FilterY/filterY.h"

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
    // Measure the speed of the x-filtering operation on a 720p
    // image with a 13-long filter.  Average over 1000 runs.

    size_t width = 1280, height = 720, len = 13, numIterations = 1000;
    bool pad = true;

    // First and second arguments: width and height
    if (argc > 2) {
        width = readStr<size_t>(argv[1]);
        height = readStr<size_t>(argv[2]);
    }

    // Third argument: filter length
    if (argc > 3) {
        len = readStr<size_t>(argv[3]);
    }

    // Fourth argument: number of iterations
    if (argc > 4) {
        numIterations = readStr<size_t>(argv[4]);
    }

    // Fifth argument: whether to perform padding
    if (argc > 5) {
        pad = readStr<bool>(argv[5]);
    }


    try {

        CLContext context;

        // Ready the command queue on the first device to hand
        cl::CommandQueue cq(context.context, context.devices[0]);

        PadX padX(context.context, context.devices);
        PadY padY(context.context, context.devices);

        std::vector<float> filter(len, 0.0);
        FilterX filterX(context.context, context.devices, filter);
        FilterY filterY(context.context, context.devices, filter);
  
        const size_t padding = 16, alignment = 16;

        // Create input and output buffers
        ImageBuffer<cl_float> input(context.context, CL_MEM_READ_WRITE,
                          width, height, padding, alignment);

        ImageBuffer<cl_float> input2(context.context, CL_MEM_READ_WRITE,
                          width, height, padding, alignment);

        ImageBuffer<cl_float> output(context.context, CL_MEM_READ_WRITE,
                           width, height, padding, alignment);

        ImageBuffer<cl_float> output2(context.context, CL_MEM_READ_WRITE,
                           width, height, padding, alignment);

        {
            // Run, timing
            auto start = std::chrono::steady_clock::now();

            for (int n = 0; n < numIterations; ++n) {
                if (pad)
                    padX(cq, input);
                filterX(cq, input, output);
            }

            cq.finish();
            auto end = std::chrono::steady_clock::now();

            // Work out what the difference between these is
            double t = DurationSeconds(end - start).count();

            std::cout << "FilterX: " 
                    << (t / numIterations * 1000) << " ms" << std::endl;
        }

        {
            // Run, timing
            auto start = std::chrono::steady_clock::now();

            for (int n = 0; n < numIterations; ++n) {
                if (pad)
                    padY(cq, input);
                filterY(cq, input, output);
            }

            cq.finish();
            auto end = std::chrono::steady_clock::now();

            // Work out what the difference between these is
            double t = DurationSeconds(end - start).count();

            std::cout << "FilterY: " 
                    << (t / numIterations * 1000) << " ms" << std::endl;

        }


        {
            // Run, timing
            auto start = std::chrono::steady_clock::now();

            for (int n = 0; n < numIterations; ++n) {
                if (pad)
                    padX(cq, input);
                filterX(cq, input, output);
                if (pad)
                    padY(cq, input2);
                filterY(cq, input2, output2);
            }

            cq.finish();
            auto end = std::chrono::steady_clock::now();

            // Work out what the difference between these is
            double t = DurationSeconds(end - start).count();

            std::cout << "Both: " 
                    << (t / numIterations * 1000) << " ms" << std::endl;

        }


    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
                     
    return 0;
}


