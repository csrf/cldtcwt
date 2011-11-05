#ifndef FILTERER_H
#define FILTERER_H

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"
#include <vector>


void colDecimateFilter(cl::Context& context,
                       cl::CommandQueue& commandQueue,
                       cl::Kernel& colDecimateFilterKernel,
                       cl::Image2D& output, cl::Image2D& input, 
                       cl::Buffer& filter, bool pad = false);

void rowDecimateFilter(cl::Context& context,
                       cl::CommandQueue& commandQueue,
                       cl::Kernel& rowDecimateFilterKernel,
                       cl::Image2D& output, cl::Image2D& input, 
                       cl::Buffer& filter, bool pad);
 
void colFilter(cl::Context& context,
               cl::CommandQueue& commandQueue,
               cl::Kernel& colFilterKernel,
               cl::Image2D& output, cl::Image2D& input, 
               cl::Buffer& filter);

class ColFilter {

public:

    ColFilter(cl::Context& context,
              const std::vector<cl::Device>& devices);

    void operator() (cl::CommandQueue& commandQueue,
               cl::Image2D& output, cl::Image2D& input, 
               cl::Buffer& filter,
               const std::vector<cl::Event>* waitEvents = 0,
               cl::Event* doneEvent = 0);

private:
    cl::Context context;
    cl::Kernel kernel;
    cl::Sampler sampler;

};


class RowFilter {

public:

    RowFilter(cl::Context& context,
              const std::vector<cl::Device>& devices);

    void operator() (cl::CommandQueue& commandQueue,
               cl::Image2D& output, cl::Image2D& input, 
               cl::Buffer& filter);

private:
    cl::Context context;
    cl::Kernel kernel;
    cl::Sampler sampler;

};

void quadToComplex(cl::Context& context,
                   cl::CommandQueue& commandQueue,
                   cl::Kernel& quadToComplexKernel,
                   cl::Image2D& out1Re, cl::Image2D& out1Im,
                   cl::Image2D& out2Re, cl::Image2D& out2Im,
                   cl::Image2D& input);

void cornernessMap(cl::Context& context,
                   cl::CommandQueue& commandQueue,
                   cl::Kernel& cornernessMapKernel,
                   cl::Image2D& output, 
                   std::vector<cl::Image2D> subbands);

cl::Image2D createImage2D(cl::Context&, cl::CommandQueue&,
                          int width, int height);
cl::Sampler createSampler(cl::Context&);

cl::Buffer createBuffer(cl::Context&, cl::CommandQueue&,
                        const float data[], int length);


#endif
