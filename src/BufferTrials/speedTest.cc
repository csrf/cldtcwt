#include <iostream>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "util/clUtil.h"

#include <sys/timeb.h>

#include "FilterX/filterX.h"



int main()
{
    // Measure the speed of the x-filtering operation on a 720p
    // image with a 13-long filter.  Average over 1000 runs.

    try {

        CLContext context;

        // Ready the command queue on the first device to hand
        cl::CommandQueue cq(context.context, context.devices[0]);

        std::vector<float> filter(13, 0.0);
        FilterX filterX(context.context, context.devices, filter);
  
        const size_t width = 1280, height = 720, 
                     padding = 8, alignment = 8;

        // Create input and output buffers
        ImageBuffer input(context.context, CL_MEM_READ_WRITE,
                          width, height, padding, alignment);

        ImageBuffer output(context.context, CL_MEM_READ_WRITE,
                           width, height, padding, alignment);


        // Run, timing
        timeb start, end;
        const int numFrames = 1000;
        ftime(&start);

        for (int n = 0; n < numFrames; ++n) {
            filterX(cq, input, output);
        }

        cq.finish();
        ftime(&end);

        // Work out what the difference between these is
        double t = end.time - start.time 
                 + 0.001 * (end.millitm - start.millitm);

        std::cout << (t / numFrames * 1000) << " ms" << std::endl;

    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
                     
    return 0;
}


