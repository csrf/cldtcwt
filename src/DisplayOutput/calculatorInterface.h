#ifndef CALCULATORINTERFACE_H
#define CALCULATORINTERFACE_H


#include "DisplayOutput/calculator.h"
#include "DisplayOutput/VBOBuffer.h"
#include "DisplayOutput/texture.h"
#include "MiscKernels/greyscaleToRGBA.h"
#include <opencv2/imgproc/imgproc.hpp>

#if defined(CL_VERSION_1_2)
    typedef cl::ImageGL GLImage;
#else
    typedef cl::Image2DGL GLImage;
#endif



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

    // Image input and CL interface
    GLTexture imageTexture_;
    GLImage imageTextureCL_;
    cl::Event imageTextureCLDone_;

    cl::Event glObjsReady_;

    // The input needs to be put into greyscale before display
    cl::Image2D imageGreyscale_;
    cl::Event imageGreyscaleDone_;


    GLuint subbandTextures_[6];

    // Where to put the keypoints
    VBOBuffers keypointLocationBuffers;




public:

    CalculatorInterface(cl::Context& context,
                        const cl::Device& device,
                        int width, int height);

    void processImage(const void* data, size_t length);

    bool isDone();

    void updateGL(void);

    GLuint getImageTexture();

};


#endif

