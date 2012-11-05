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
   imageTexture_(GL_RGBA8, width, height),
   imageTextureCL_(context, CL_MEM_READ_WRITE, 
                   GL_TEXTURE_2D, 0, imageTexture_.getTexture()),
   imageGreyscale_(context, CL_MEM_READ_WRITE, 
                   cl::ImageFormat(CL_LUMINANCE, CL_UNORM_INT8),
                   width, height),
   calculator_(context, device, width, height),
   pboBuffer_(1)
{
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

    std::vector<cl::Memory> glTransferObjs = {imageTextureCL_};

    cl::Event glObjsAcquired;
    cq_.enqueueAcquireGLObjects(&glTransferObjs, nullptr, &glObjsAcquired);

    greyscaleToRGBA_(cq_, imageGreyscale_, imageTextureCL_,
                     {imageGreyscaleDone_, glObjsAcquired}, 
                     &imageTextureCLDone_);

    std::vector<cl::Event> releaseEvents = {imageTextureCLDone_};
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

