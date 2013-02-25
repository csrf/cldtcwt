#ifndef CALCULATORINTERFACE_H
#define CALCULATORINTERFACE_H

#include "Filter/imageBuffer.h"

#include "calculator.h"
#include "VBOBuffer.h"
#include "texture.h"
#include "GreyscaleToRGBA/greyscaleToRGBA.h"
#include "AbsToRGBA/absToRGBA.h"
#include <array>

#include "Filter/ImageToImageBuffer/imageToImageBuffer.h"

#if defined(CL_VERSION_1_2)
    typedef cl::ImageGL GLImage;
#else
    typedef cl::Image2DGL GLImage;
#endif


// Number of subbands the DTCWT produces
const size_t numSubbands = 6;

class CalculatorInterface {

private:
    unsigned int width_, height_;

    Calculator calculator_;

    // For interop OpenGL/OpenCL

    VBOBuffers pboBuffer_;
    // Used for quickly transfering the image

    cl::CommandQueue cq_;
    // Used for upload and output conversion

    // Kernel to convert into RGBA for display
    GreyscaleToRGBA greyscaleToRGBA_;
    AbsToRGBA absToRGBA_;

    // Image input and CL interface
    GLTexture imageTexture_;
    GLImage imageTextureCL_;
    cl::Event imageTextureCLDone_;


    // Energy map and CL interface
    GLTexture energyMapTexture_;
    GLImage energyMapTextureCL_;
    cl::Event energyMapTextureCLDone_;

    // Done when everything is copied over to the GL objects
    cl::Event glObjsReady_;

    // The input needs to be put into greyscale before display
    cl::Image2D imageGreyscale_;
    cl::Event imageGreyscaleDone_;

    // To convert above into below:
    ImageToImageBuffer imageToImageBuffer_;

    // And also copied into a buffer for the DTCWT input
    ImageBuffer<cl_float> bufferGreyscale_;
    cl::Event bufferGreyscaleDone_;

    // For subband displays for levels 2 and 3
    std::array<GLTexture, numSubbands> subbandTextures2_;
    std::array<GLImage, numSubbands> subbandTextures2CL_;

    std::array<GLTexture, numSubbands> subbandTextures3_;
    std::array<GLImage, numSubbands> subbandTextures3CL_;

    // Where to put the keypoints
    VBOBuffers keypointLocationsBuffer_;
    cl::BufferGL keypointLocationsBufferCL_;
    cl::Event kpLocsCopied_;


public:

    CalculatorInterface(cl::Context& context,
                        const cl::Device& device,
                        int width, int height);

    void processImage(const void* data, size_t length);

    bool isDone();
    void waitUntilDone();

    void updateGL(void);

    GLuint getImageTexture();
    GLuint getEnergyMapTexture();
    GLuint getSubband2Texture(int subband);
    GLuint getSubband3Texture(int subband);

    GLuint getKeypointLocations();
    size_t getNumKeypointLocations();
    size_t getNumFloatsPerKeypointLocation();

};


#endif

