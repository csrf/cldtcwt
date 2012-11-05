#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include "util/clUtil.h"


#include "DisplayOutput/calculatorInterface.h"
#include <iostream>


CalculatorInterface::CalculatorInterface(cl::Context& context,
                                         const cl::Device& device,
                                         int width, int height)
 : width_(width), height_(height),
   cq_(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE),
   greyscaleToRGBA_(context, {device}),
   absToRGBA_(context, {device}),
   imageTexture_(GL_RGBA8, width, height),
   imageTextureCL_(context, CL_MEM_READ_WRITE, 
                   GL_TEXTURE_2D, 0, imageTexture_.getTexture()),
   imageGreyscale_(context, CL_MEM_READ_WRITE, 
                   cl::ImageFormat(CL_LUMINANCE, CL_UNORM_INT8),
                   width, height),
   calculator_(context, device, width, height),
   pboBuffer_(1)
{
    // Set up the subband textures
    for (size_t n = 0; n < subbandTextures_.size(); ++n) {

        // Create OpenGL texture
        subbandTextures_[n] = GLTexture(GL_RGBA8, width / 4, height / 4);

        // Add OpenCL link to it
        subbandTexturesCL_[n]
            = GLImage(context, CL_MEM_READ_WRITE, 
                      GL_TEXTURE_2D, 0, 
                      subbandTextures_[n].getTexture());

    }
}

#include <cstring>

void CalculatorInterface::processImage(const void* data, size_t length)
{
    // Upload using OpenCL, not copying the data into its own memory.  This
    // means we can't use the data until the transfer is done.
    cq_.enqueueWriteImage(imageGreyscale_, 
                          // Don't block
                          CL_FALSE, 
                          // Start corner and size
                          makeCLSizeT<3>({0, 0, 0}), 
                          makeCLSizeT<3>({width_, height_, 1}), 
                          // Stride and data pointer
                          0, 0, data,
                          nullptr, &imageGreyscaleDone_);

    calculator_(imageGreyscale_, {imageGreyscaleDone_});

    // Go over to using the OpenGL objects.  glFinish should already have
    // been called
    std::vector<cl::Memory> glTransferObjs = {imageTextureCL_};
    std::copy(subbandTexturesCL_.begin(), subbandTexturesCL_.end(), 
              std::back_inserter(glTransferObjs));

    cl::Event glObjsAcquired;
    cq_.enqueueAcquireGLObjects(&glTransferObjs, nullptr, &glObjsAcquired);

    // Convert the input image to RGBA for display
    greyscaleToRGBA_(cq_, imageGreyscale_, imageTextureCL_,
                     {imageGreyscaleDone_, glObjsAcquired}, 
                     &imageTextureCLDone_);

    auto subbands = calculator_.levelOutputs();

    std::array<cl::Event, numSubbands> subbandsConverted;

    // Convert the subbands to absolute images

    // Wait for the level and the GL objects to be acquired
    std::vector<cl::Event> subbandsInputReady = {glObjsAcquired};
    std::copy(subbands[0]->done.begin(), subbands[0]->done.end(),
              std::back_inserter(subbandsInputReady));
    
    for (size_t n = 0; n < numSubbands; ++n) 
        absToRGBA_(cq_, subbands[0]->sb[n], subbandTexturesCL_[n], 4.0f,
                        subbandsInputReady, &subbandsConverted[n]);

    // Stop using the OpenGL objects
    std::vector<cl::Event> releaseEvents = {imageTextureCLDone_};
    std::copy(subbandsConverted.begin(), subbandsConverted.end(),
              std::back_inserter(releaseEvents));

    cq_.enqueueReleaseGLObjects(&glTransferObjs,
                                &releaseEvents, &glObjsReady_);
}


bool CalculatorInterface::isDone()
{
    return glObjsReady_.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>()
                == CL_COMPLETE;
}


void CalculatorInterface::updateGL(void)
{
    // TODO
}



GLuint CalculatorInterface::getImageTexture()
{
    return imageTexture_.getTexture();
}



GLuint CalculatorInterface::getSubbandTexture(int subband)
{
    return subbandTextures_[subband].getTexture();
}

