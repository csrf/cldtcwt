#include <iostream>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "util/clUtil.h"

#include <chrono>
typedef std::chrono::duration<double>
    DurationSeconds;

#include "decimateFilterX.h"
#include "../PadX/padX.h"



int main()
{
    // Measure the speed of the decimated x-filtering operation on a 720p
    // image with a 14-long filter.  Average over 1000 runs.

    try {

        CLContext context;

        // Ready the command queue on the first device to hand
        cl::CommandQueue cq(context.context, context.devices[0]);

        std::vector<float> filter(14, 0.0);
        DecimateFilterX filterX(context.context, context.devices, filter,
                                false);
        PadX padX(context.context, context.devices);
  
        const size_t width = 1280, height = 720, 
                     padding = 16, alignment = 2*16;

        // Create input and output buffers
        ImageBuffer<cl_float> input(context.context, CL_MEM_READ_WRITE,
                          width, height, padding, alignment);

        ImageBuffer<cl_float> output(context.context, CL_MEM_READ_WRITE,
                           width / 2, height, padding, alignment);

        {
            // Run, timing
            const int numFrames = 1000;
            auto start = std::chrono::steady_clock::now();

            for (int n = 0; n < numFrames; ++n) {
                padX(cq, input);
                filterX(cq, input, output);
            }

            cq.finish();
            auto end = std::chrono::steady_clock::now();

            // Work out what the difference between these is
            double t = DurationSeconds(end - start).count();

            std::cout << "DecimateFilterX: " 
                    << (t / numFrames * 1000) << " ms" << std::endl;
        }
    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
                     
    return 0;
}


