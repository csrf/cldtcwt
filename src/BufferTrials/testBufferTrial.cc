#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <algorithm>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "util/clUtil.h"

#include <sys/timeb.h>

#include "filter.h"

#include <Eigen/Dense>


unsigned int wrap(int n, int width)
{
    // Wrap so that the pattern goes
    // forwards-backwards-forwards-backwards etc, with the end
    // values repeated.
    
    unsigned int result = n % (2 * width);

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
    try {

        CLContext context;

        // Ready the command queue on the first device to hand
        cl::CommandQueue cq(context.context, context.devices[0]);

        FilterX filterX(context.context, context.devices, filter);

  
        const size_t width = 1280, height = 720, 
                     padding = 8, alignment = 8;

        ImageBuffer input(context.context, CL_MEM_READ_WRITE,
                          width, height, padding, alignment); 

        ImageBuffer output(context.context, CL_MEM_READ_WRITE,
                           width, height, padding, alignment); 


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
        throw;
    }

}



int main()
{

    std::vector<float> filter(13, 0.0);
    Eigen::ArrayXXf X(20,10);
    
   

                         
    return 0;
}


