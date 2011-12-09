#ifndef FILTERER_H
#define FILTERER_H

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"
#include <vector>


class ColDecimateFilter {
    // Class that provides column filtering capabilities

public:

    ColDecimateFilter(cl::Context& context,
              const std::vector<cl::Device>& devices);

    void operator() (cl::CommandQueue& commandQueue,
               cl::Image2D& output, cl::Image2D& input, 
               cl::Buffer& filter,
               bool pad = false,
               const std::vector<cl::Event>* waitEvents = 0,
               cl::Event* doneEvent = 0);

private:
    cl::Context context;
    cl::Kernel kernel;
    cl::Sampler sampler;

};



class RowDecimateFilter {
    // Class that provides row decimated filtering capabilities

public:

    RowDecimateFilter(cl::Context& context,
              const std::vector<cl::Device>& devices);

    void operator() (cl::CommandQueue& commandQueue,
               cl::Image2D& output, cl::Image2D& input, 
               cl::Buffer& filter,
               bool pad = false,
               const std::vector<cl::Event>* waitEvents = 0,
               cl::Event* doneEvent = 0);

private:
    cl::Context context;
    cl::Kernel kernel;
    cl::Sampler sampler;

};


class ColFilter {
    // Class that provides column filtering capabilities

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
    // Class that provides row filtering capabilities

public:

    RowFilter(cl::Context& context,
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


class QuadToComplex {
    // Class that converts an interleaved image to two subbands with real
    // and imaginary components

public:

    QuadToComplex(cl::Context& context,
              const std::vector<cl::Device>& devices);

    void operator() (cl::CommandQueue& commandQueue,
               cl::Image2D& out1Re, cl::Image2D& out1Im,
               cl::Image2D& out2Re, cl::Image2D& out2Im,
               cl::Image2D& input,
               const std::vector<cl::Event>* waitEvents = 0,
               cl::Event* doneEvent = 0);

private:
    cl::Context context;
    cl::Kernel kernel;
    cl::Sampler sampler;

};


void cornernessMap(cl::Context& context,
                   cl::CommandQueue& commandQueue,
                   cl::Kernel& cornernessMapKernel,
                   cl::Image2D& output, 
                   std::vector<cl::Image2D> subbands);


cl::Buffer createBuffer(cl::Context&, cl::CommandQueue&,
                        const std::vector<float>& data);

cl::Image2D createImage2D(cl::Context&, int width, int height);

void writeImage2D(cl::CommandQueue& commandQueue,
                  cl::Image2D& image, float* memory);

void readImage2D(cl::CommandQueue& commandQueue,
                 float* outMemory, cl::Image2D& image);

#endif
