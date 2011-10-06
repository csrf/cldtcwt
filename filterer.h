#ifndef FILTERER_H
#define FILTERER_H

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"
#include "cv.h"
#include <vector>

class Filterer {
public:
    Filterer();

    void colDecimateFilter(cl::Image2D& output, cl::Image2D& input, 
                           cl::Buffer& filter, bool pad = false);
    void rowDecimateFilter(cl::Image2D& output, cl::Image2D& input, 
                           cl::Buffer& filter, bool pad = false);

    void colFilter(cl::Image2D& output, cl::Image2D& input, 
                           cl::Buffer& filter);
    void rowFilter(cl::Image2D& output, cl::Image2D& input, 
                           cl::Buffer& filter);

    void quadToComplex(cl::Image2D& out1Re, cl::Image2D& out1Im,
                       cl::Image2D& out2Re, cl::Image2D& out2Im,
                       cl::Image2D& input);

    void cornernessMap(cl::Image2D& output, std::vector<cl::Image2D> subbands);

    cl::Image2D createImage2D(cv::Mat& image);
    cl::Image2D createImage2D(int width, int height);
    cv::Mat getImage2D(cl::Image2D&);
    cl::Sampler createSampler();

    cl::Buffer createBuffer(const float data[], int length);

private:

    cl::Context context;
    cl::Program program;
    cl::CommandQueue commandQueue;

    cl::Kernel rowDecimateFilterKernel;
    cl::Kernel colDecimateFilterKernel;
    cl::Kernel rowFilterKernel;
    cl::Kernel colFilterKernel;
    cl::Kernel quadToComplexKernel;
    cl::Kernel cornernessMapKernel;
};



#endif
