#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

#include "filterer.h"
#include "clUtil.h"
#include "dtcwt.h"
#include <iomanip>

#include <ctime>

#include <stdexcept>

#include <highgui.h>



int main()
{
    try {

        CLContext context;

        // Ready the command queue on the first device to hand
        cl::CommandQueue cq(context.context, context.devices[0]);

        //-----------------------------------------------------------------
        // Starting test code
  
        cv::Mat input = cv::Mat::zeros(32, 4, cv::DataType<float>::type);
        input.at<float>(16,3) = 1.0f;
        cl::Image2D inImage = createImage2D(context.context, input);

        Filter h = { 
            context.context, context.devices, 
            createBuffer(context.context, cq, {0.5, 1, 0.5}),
            Filter::y 
        };

        cl::Image2D outImage
            = createImage2D(context.context, 
                                     inImage.getImageInfo<CL_IMAGE_WIDTH>(),
                                     inImage.getImageInfo<CL_IMAGE_HEIGHT>());

        h(cq, inImage, outImage);


        DecimateFilter hd = { 
            context.context, context.devices, 
            createBuffer(context.context, cq, {0.5, 0.0, 1.0, 0.5}),
            DecimateFilter::y 
        };

        cl::Image2D outImageD
            = createImage2D(context.context, 
                            inImage.getImageInfo<CL_IMAGE_WIDTH>(),
                            inImage.getImageInfo<CL_IMAGE_HEIGHT>() / 2);

        hd(cq, inImage, outImageD);

        cq.finish();

        displayRealImage(cq, outImage);
        displayRealImage(cq, outImageD);

    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
                     
    return 0;
}



