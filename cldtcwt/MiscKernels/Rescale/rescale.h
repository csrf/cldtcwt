// Copyright (C) 2013 Timothy Gale
#ifndef RESCALE_H
#define RESCALE_H



#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"



class Rescale {
    // Class that takes an image, and interpolates it up or down to a new
    // scale.  The scaling is centred around the centre point of the
    // image
    
public:

    Rescale(cl::Context& context, const std::vector<cl::Device>& devices);

    void operator() (cl::CommandQueue& commandQueue,
                     cl::Image& input,
                     cl::Image2D& output,
                     float scalingFactor,
                     const std::vector<cl::Event>& waitEvents
                         = std::vector<cl::Event>(),
                     cl::Event* doneEvent = nullptr);


private:

    cl::Context context_;
    cl::Kernel kernel_;

};


#endif

