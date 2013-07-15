// Copyright (C) 2013 Timothy Gale
#ifndef FOUR_TREE_H
#define FOUR_TREE_H

#include "DTCWT/dtcwt.h"
#include "DTCWT/filterer.h"
#include <array>


struct FourTreeTemps {

    std::array<cl::Image2D, 3> scaledInput;
    std::array<DtcwtTemps, 4> temps;

};


struct FourTreeOutputs {

    // One per tree
    std::array<DtcwtOutput, 4> outputs;

    FourTreeOutputs(const FourTreeOutputs&) = default;
    FourTreeOutputs() = default;

    // Creates the images to hold the outputs, given the inputs
    FourTreeOutputs(const FourTreeTemps& temps);

};


class FourTree {

private:

    Dtcwt dtcwt_;
    cl::Context context_;
    Rescale rescale_;
    
public:

    FourTree() = default;
    FourTree(const FourTree&) = default;

    FourTree(cl::Context& context, 
             const std::vector<cl::Device>& devices,
             cl::CommandQueue commandQueue,
             float scaleFactor = 1.f);
             
    // Create the temporary images needed to process an input of the given size
    FourTreeTemps createTemps(size_t imageWidth, size_t imageHeight,
                              size_t numLevels, size_t startLevel);
    
    // Process an image with all four trees
    void operator() (cl::CommandQueue& commandQueue,
                     cl::Image& image,
                     FourTreeTemps& temps,
                     FourTreeOutputs& treeOutputs,
                     const std::vector<cl::Event>& waitEvents
                        = std::vector<cl::Event>());



};

#endif

