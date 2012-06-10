#ifndef FILTERER_H
#define FILTERER_H

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "cl.hpp"
#include <vector>


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

private:
    cl::Context context_;
    cl::Kernel kernel_;
    cl::Buffer coefficients_;
    const Direction dimension_;

    const int wgSizeX_;
    const int wgSizeY_;
};



class DecimateFilter {
// Class that provides decimation filtering capabilities

public:

    enum Direction { x, y };
    DecimateFilter(cl::Context& context,
                   const std::vector<cl::Device>& devices,
                   cl::Buffer coefficients,
                   Direction d,
                   bool swapTrees = false);
    // If swapTrees is enabled, when outputing the two trees being rpoduced
    // are swapped over.  Had not read about this before, but coldfilt does
    // it (strangely not mentioning it, so far as I could see, in its
    // description).

    // The filter operation
    void operator() (cl::CommandQueue& commandQueue,
           const cl::Image2D& input,
           cl::Image2D& output,
           const std::vector<cl::Event>& waitEvents = std::vector<cl::Event>(),
           cl::Event* doneEvent = nullptr);


private:
    cl::Context context_;
    cl::Kernel kernel_;
    cl::Buffer coefficients_;
    const Direction dimension_;

    const int wgSizeX_;
    const int wgSizeY_;
};




class QuadToComplex {
    // Class that converts an interleaved image to two subbands with real
    // and imaginary components

public:

    QuadToComplex(cl::Context& context,
              const std::vector<cl::Device>& devices);

    void
    operator() (cl::CommandQueue& commandQueue,
           const cl::Image2D& input,
           cl::Image2D& out1, cl::Image2D& out2,
           const std::vector<cl::Event>& waitEvents = std::vector<cl::Event>(),
           cl::Event* doneEvent = nullptr);

private:
    cl::Context context;
    cl::Kernel kernel;
    cl::Sampler sampler;

};


class Rescale {
    // Class that takes an image, and interpolates it up or down to a new
    // scale.  The scaling is centred around the centre point of the
    // image
    
public:

    Rescale(cl::Context& context, const std::vector<cl::Device>& devices);

    void operator() (cl::CommandQueue& commandQueue,
                     const cl::Image2D& input,
                     cl::Image2D& output,
                     float scalingFactor,
                     const std::vector<cl::Event>& waitEvents
                         = std::vector<cl::Event>(),
                     cl::Event* doneEvent = nullptr);


private:

    cl::Context context_;
    cl::Kernel kernel_;

};




void cornernessMap(cl::Context& context,
                   cl::CommandQueue& commandQueue,
                   cl::Kernel& cornernessMapKernel,
                   cl::Image2D& output, 
                   std::vector<cl::Image2D> subbands);

#endif
