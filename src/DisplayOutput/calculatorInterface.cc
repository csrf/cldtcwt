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
   calculator_(context, device, width, height),
   cq_(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE),
   greyscaleToRGBA_(context, {device}),
   absToRGBA_(context, {device}),
   imageTexture_(GL_RGBA8, width, height),
   imageTextureCL_(context, CL_MEM_READ_WRITE, 
                   GL_TEXTURE_2D, 0, imageTexture_.getTexture()),
   imageGreyscale_(context, CL_MEM_READ_WRITE, 
                   cl::ImageFormat(CL_LUMINANCE, CL_UNORM_INT8),
                   width, height),
   pboBuffer_(1),
   keypointLocationsBuffer_(1)
{
    // Set up the subband textures
    for (size_t n = 0; n < numSubbands; ++n) {

        // Create OpenGL texture
        subbandTextures2_[n] = GLTexture(GL_RGBA8, width / 4, height / 4);

        // Add OpenCL link to it
        subbandTextures2CL_[n]
            = GLImage(context, CL_MEM_READ_WRITE, 
                      GL_TEXTURE_2D, 0, 
                      subbandTextures2_[n].getTexture());

        // Create OpenGL texture
        subbandTextures3_[n] = GLTexture(GL_RGBA8, width / 8, height / 8);

        // Add OpenCL link to it
        subbandTextures3CL_[n]
            = GLImage(context, CL_MEM_READ_WRITE, 
                      GL_TEXTURE_2D, 0, 
                      subbandTextures3_[n].getTexture());


    }

    // Set up for the energy map texture
    energyMapTexture_ = GLTexture(GL_RGBA8, 
        calculator_.getEnergyMapLevel2().getImageInfo<CL_IMAGE_WIDTH>(),
        calculator_.getEnergyMapLevel2().getImageInfo<CL_IMAGE_HEIGHT>());

    // Add OpenCL link to it
    energyMapTextureCL_ = GLImage(context, CL_MEM_READ_WRITE, 
                                  GL_TEXTURE_2D, 0, 
                                  energyMapTexture_.getTexture());

    // Set up the keypoints location buffer
    glBindBuffer(GL_ARRAY_BUFFER, keypointLocationsBuffer_.getBuffer(0));
    glBufferData(GL_ARRAY_BUFFER, 
                 calculator_.keypointLocations().getInfo<CL_MEM_SIZE>(), 
                 nullptr, // No need to actually upload any data
                 GL_DYNAMIC_DRAW);

    // Add its OpenCL link
    keypointLocationsBufferCL_ 
        = cl::BufferGL(context, CL_MEM_READ_WRITE,
                                  keypointLocationsBuffer_.getBuffer(0));
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
    std::vector<cl::Memory> glTransferObjs = {imageTextureCL_,
                                              energyMapTextureCL_,
                                              keypointLocationsBufferCL_};
    std::copy(subbandTextures2CL_.begin(), subbandTextures2CL_.end(), 
              std::back_inserter(glTransferObjs));
    std::copy(subbandTextures3CL_.begin(), subbandTextures3CL_.end(), 
              std::back_inserter(glTransferObjs));

    cl::Event glObjsAcquired;
    cq_.enqueueAcquireGLObjects(&glTransferObjs, nullptr, &glObjsAcquired);

    // Convert the input image to RGBA for display
    greyscaleToRGBA_(cq_, imageGreyscale_, imageTextureCL_, 1.0f,
                     {imageGreyscaleDone_, glObjsAcquired}, 
                     &imageTextureCLDone_);

    auto subbands = calculator_.levelOutputs();

    std::vector<cl::Event> subbandsConverted(2*numSubbands);

    // Convert the subbands to absolute images

    // Wait for the level and the GL objects to be acquired
    std::vector<cl::Event> subbandsInput2Ready = {glObjsAcquired};
    std::copy(subbands[0]->done.begin(), subbands[0]->done.end(),
              std::back_inserter(subbandsInput2Ready));

    std::vector<cl::Event> subbandsInput3Ready = {glObjsAcquired};
    std::copy(subbands[1]->done.begin(), subbands[1]->done.end(),
              std::back_inserter(subbandsInput3Ready));
    
    for (size_t n = 0; n < numSubbands; ++n) {

        absToRGBA_(cq_, subbands[0]->sb[n], 
                        subbandTextures2CL_[n], 4.0f, subbandsInput2Ready, 
                        &subbandsConverted[n]);

        absToRGBA_(cq_, subbands[1]->sb[n], 
                        subbandTextures3CL_[n], 4.0f, subbandsInput3Ready, 
                        &subbandsConverted[numSubbands+n]);

    }

    std::vector<cl::Event> energyMapReady = calculator_.keypointLocationEvents();
    energyMapReady.push_back(glObjsAcquired);

    cl::Image2D energyMapInput = calculator_.getEnergyMapLevel2();
    // Convert the energy map
    greyscaleToRGBA_(cq_, energyMapInput,
                          energyMapTextureCL_,
                          1.f,
                          energyMapReady, &energyMapTextureCLDone_);

    // Copy the keypoint locations over
    std::vector<cl::Event> kplEvents = calculator_.keypointLocationEvents();
    cq_.enqueueCopyBuffer(calculator_.keypointLocations(), 
                          keypointLocationsBufferCL_, 
                      0, 0, keypointLocationsBufferCL_.getInfo<CL_MEM_SIZE>(),
                          &kplEvents,
                          &kpLocsCopied_);

    // Stop using the OpenGL objects
    std::vector<cl::Event> releaseEvents = {imageTextureCLDone_,
                                            energyMapTextureCLDone_,
                                            kpLocsCopied_};
    std::copy(subbandsConverted.begin(), subbandsConverted.end(),
              std::back_inserter(releaseEvents));

    cq_.enqueueReleaseGLObjects(&glTransferObjs,
                                &releaseEvents, &glObjsReady_);
}


bool CalculatorInterface::isDone()
{
    // Warning: this seems to complete much too early.  Strange.
    return glObjsReady_.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>()
                == CL_COMPLETE;
}


void CalculatorInterface::waitUntilDone()
{
    cq_.finish();
}



void CalculatorInterface::updateGL(void)
{
    // TODO
}



GLuint CalculatorInterface::getImageTexture()
{
    return imageTexture_.getTexture();
}



GLuint CalculatorInterface::getEnergyMapTexture()
{
    return energyMapTexture_.getTexture();
}



GLuint CalculatorInterface::getSubband2Texture(int subband)
{
    return subbandTextures2_[subband].getTexture();
}



GLuint CalculatorInterface::getSubband3Texture(int subband)
{
    return subbandTextures3_[subband].getTexture();
}


GLuint CalculatorInterface::getKeypointLocations()
{
    // Gets the buffer containing a list of keypoint locations
    
    return keypointLocationsBuffer_.getBuffer(0);
}


size_t CalculatorInterface::getNumKeypointLocations()
{
    // Read the number of keypoint locations.  Note, involves a transfer
    // from the graphics card so a little more expensive than some other
    // ops.
    cl::Buffer kplCumSum = calculator_.keypointCumCounts();

    // It's a cumulative sum, so the total is in the last element
    size_t position = kplCumSum.getInfo<CL_MEM_SIZE>() - sizeof(cl_uint);

    std::vector<cl::Event> waitEvents = calculator_.keypointLocationEvents();

    cl_uint val;
    cq_.enqueueReadBuffer(kplCumSum, CL_TRUE, 
                          position, sizeof(cl_uint), &val,
                          &waitEvents);

    return val;
}


size_t CalculatorInterface::getNumFloatsPerKeypointLocation()
{
    // For the list of keypoint locations, returns how many floating points
    // each keypoint entry contains.  The format is (x, y, scale), but
    // might then be padded out.  x and y are relative to the centre of the
    // image; scale is the radius of the keypoint.

    return calculator_.numFloatsPerKPLocation();
}

