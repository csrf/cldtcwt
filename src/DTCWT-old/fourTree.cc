// Copyright (C) 2013 Timothy Gale
#include "DTCWT/fourTree.h"
#include "util/clUtil.h"

// Initialisation of four tree outputs by initialising each tree from its
// temporaries
FourTreeOutputs::FourTreeOutputs(const FourTreeTemps& temps)
{
    const size_t numTrees = 4;

    // Create the outputs for each tree
    for (int n = 0; n < numTrees; ++n)
        outputs[n] = DtcwtOutput(temps.temps[n]);

}


FourTreeTemps FourTree::createTemps(size_t imageWidth, size_t imageHeight,
                                    size_t numLevels, size_t startLevel)
{
    FourTreeTemps temps;

    // Create the images to put scaled versions of the input into
    temps.scaledInput[0] = createImage2D(context_, imageWidth * 7 / 8, 
                                                  imageHeight * 7 / 8);
    temps.scaledInput[1] = createImage2D(context_, imageWidth * 6 / 8, 
                                                  imageHeight * 6 / 8);
    temps.scaledInput[2] = createImage2D(context_, imageWidth * 5 / 8, 
                                                  imageHeight * 5 / 8);

    // Create the DTCWT temporary value holders
    temps.temps[0] = dtcwt_.createContext(imageWidth, imageHeight,
                                          numLevels, startLevel);
    temps.temps[1] = 
        dtcwt_.createContext(temps.scaledInput[0].getImageInfo<CL_IMAGE_WIDTH>(), 
                             temps.scaledInput[0].getImageInfo<CL_IMAGE_HEIGHT>(), 
                             numLevels, startLevel);
    temps.temps[2] = 
        dtcwt_.createContext(temps.scaledInput[1].getImageInfo<CL_IMAGE_WIDTH>(), 
                             temps.scaledInput[1].getImageInfo<CL_IMAGE_HEIGHT>(), 
                             numLevels, startLevel);
    temps.temps[3] = 
        dtcwt_.createContext(temps.scaledInput[2].getImageInfo<CL_IMAGE_WIDTH>(), 
                             temps.scaledInput[2].getImageInfo<CL_IMAGE_HEIGHT>(), 
                             numLevels, startLevel);

    return temps;
}
    


FourTree::FourTree(cl::Context& context, 
                   const std::vector<cl::Device>& devices,
                   cl::CommandQueue commandQueue,
                   float scaleFactor)
    : context_(context),
      dtcwt_(context, devices, commandQueue, scaleFactor),
      rescale_(context, devices)
{
}


void FourTree::operator() (cl::CommandQueue& commandQueue,
                     cl::Image& image,
                     FourTreeTemps& temps,
                     FourTreeOutputs& treeOutputs,
                     const std::vector<cl::Event>& waitEvents)
{
    // First tree
    dtcwt_(commandQueue, image, temps.temps[0], treeOutputs.outputs[0],
           waitEvents);

    // Second tree (with downscaling)
    cl::Event scaleOneDone;
    rescale_(commandQueue, image, temps.scaledInput[0], 
             7.0f / 8.0f, waitEvents, &scaleOneDone);
    dtcwt_(commandQueue, 
           temps.scaledInput[0], temps.temps[1], treeOutputs.outputs[1],
           {scaleOneDone});

    // Third tree (with downscaling)
    cl::Event scaleTwoDone;
    rescale_(commandQueue, image, temps.scaledInput[1], 
             6.0f / 8.0f, waitEvents, &scaleTwoDone);
    dtcwt_(commandQueue, 
           temps.scaledInput[1], temps.temps[2], treeOutputs.outputs[2],
           {scaleTwoDone});

    // Fourth tree (with downscaling)
    cl::Event scaleThreeDone;
    rescale_(commandQueue, image, temps.scaledInput[2], 
             5.0f / 8.0f, waitEvents, &scaleThreeDone);
    dtcwt_(commandQueue, 
           temps.scaledInput[2], temps.temps[3], treeOutputs.outputs[3],
           {scaleThreeDone});
}


