#ifndef FILTERER_H
#define FILTERER_H

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "cl.hpp"
#include <vector>
#include <tuple>


class ColDecimateFilter {
    // Class that provides column filtering capabilities

public:

    ColDecimateFilter(cl::Context& context,
              const std::vector<cl::Device>& devices);

    cl::Image2D operator() (cl::CommandQueue& commandQueue,
         cl::Image2D& input, cl::Buffer& filter,
         const std::vector<cl::Event>& waitEvents = std::vector<cl::Event>(),
         cl::Event* doneEvent = nullptr,
         cl::Image2D* targetImage = nullptr);

    // Create an image with the right size to hold the output
    cl::Image2D dummyRun(const cl::Image2D& input);
    cl::Image2D dummyRun(size_t inWidth, size_t inHeight);

private:
    cl::Kernel kernel;
    cl::Context context;
    cl::Sampler sampler;

};



class RowDecimateFilter {
    // Class that provides row decimated filtering capabilities

public:

    RowDecimateFilter(cl::Context& context,
              const std::vector<cl::Device>& devices);

    cl::Image2D operator() (cl::CommandQueue& commandQueue,
           cl::Image2D& input, cl::Buffer& filter,
           const std::vector<cl::Event>& waitEvents = std::vector<cl::Event>(),
           cl::Event* doneEvent = nullptr,
           cl::Image2D* targetImage = nullptr);

    // Create an image with the right size to hold the output
    cl::Image2D dummyRun(const cl::Image2D& input);
    cl::Image2D dummyRun(size_t inWidth, size_t inHeight);

private:
    cl::Context context;
    cl::Kernel kernel;
    cl::Sampler sampler;

};




class Filter {
// Class that provides filtering capabilities

public:

    enum Direction { x, y };
    Filter(cl::Context& context,
           const std::vector<cl::Device>& devices,
           cl::Buffer coefficients,
           Direction d);

    // The filter operation
    void operator() (cl::CommandQueue& commandQueue,
           const cl::Image2D& input,
           cl::Image2D& output,
           const std::vector<cl::Event>& waitEvents = std::vector<cl::Event>(),
           cl::Event* doneEvent = nullptr);

    // Create an image with the right size to hold the output
    cl::Image2D dummyRun(const cl::Image2D& input);
    cl::Image2D dummyRun(size_t inWidth, size_t inHeight);


private:
    cl::Context context_;
    cl::Kernel kernel_;
    cl::Buffer coefficients_;
    const Direction dimension_;

    const int wgSizeX_;
    const int wgSizeY_;
};



class ColFilter {
    // Class that provides column filtering capabilities

public:

    ColFilter(cl::Context& context,
              const std::vector<cl::Device>& devices);

    // The filter operation
    cl::Image2D operator() (cl::CommandQueue& commandQueue,
           cl::Image2D& input, cl::Buffer& filter,
           const std::vector<cl::Event>& waitEvents = std::vector<cl::Event>(),
           cl::Event* doneEvent = nullptr,
           cl::Image2D* targetImage = nullptr);

    // Create an image with the right size to hold the output
    cl::Image2D dummyRun(const cl::Image2D& input);
    cl::Image2D dummyRun(size_t inWidth, size_t inHeight);

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

    cl::Image2D operator() (cl::CommandQueue& commandQueue,
           cl::Image2D& input, cl::Buffer& filter,
           const std::vector<cl::Event>& waitEvents = std::vector<cl::Event>(),
           cl::Event* doneEvent = nullptr,
           cl::Image2D* targetImage = nullptr);

    // Create an image with the right size to hold the output
    cl::Image2D dummyRun(const cl::Image2D& input);
    cl::Image2D dummyRun(size_t inWidth, size_t inHeight);

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

    std::tuple<cl::Image2D, cl::Image2D>
    operator() (cl::CommandQueue& commandQueue,
           cl::Image2D& input,
           const std::vector<cl::Event>& waitEvents = std::vector<cl::Event>(),
           cl::Event* doneEvent = nullptr,
           cl::Image2D* target1 = nullptr,
           cl::Image2D* target2 = nullptr);

    // Create an image with the right size to hold the output
    cl::Image2D dummyRun(const cl::Image2D& input);
    cl::Image2D dummyRun(size_t inWidth, size_t inHeight);

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

#endif
