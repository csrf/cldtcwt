#ifndef CALCULATORINTERFACE_H
#define CALCULATORINTERFACE_H


#include "DisplayOutput/calculator.h"
#include "DisplayOutput/VBOBuffer.h"
#include <opencv2/imgproc/imgproc.hpp>


class CalculatorInterface {

private:
    Calculator calculator_;

    // For interop OpenGL/OpenCL
    GLuint imageTexture_;
    GLuint subbandTextures_[6];

    // Where to put the keypoints
    VBOBuffers keypointLocationBuffers;

public:

    CalculatorInterface(cl::Context& context,
                        const cl::Device& device,
                        int width, int height);

    void processImage(cv::Mat input);

    void updateGL(void);



};


#endif

