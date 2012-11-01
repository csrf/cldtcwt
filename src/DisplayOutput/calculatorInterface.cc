#include "DisplayOutput/calculatorInterface.h"
#include <iostream>


CalculatorInterface::CalculatorInterface(cl::Context& context,
                                         const cl::Device& device,
                                         int width, int height)
 : imageTexture_(GL_RGBA8, width, height),
   calculator_(context, device, width, height)
{
    // Couple it for CL to use
    imageTextureCL_ 
        = GLImage(context, CL_MEM_READ_WRITE, 
                  GL_TEXTURE_2D, 0, imageTexture_.getTexture());


}


void CalculatorInterface::processImage(const cv::Mat& input)
{
    // Take the image in BGR unsigned byte format

    // Upload the texture
	glBindTexture(GL_TEXTURE_2D, imageTexture_.getTexture());
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
    			 input.cols, input.rows, 
    			 GL_BGR, GL_UNSIGNED_BYTE,  // Input format
                 input.data);

    // Make sure everything's done before we acquire for use by OpenCL
    glFinish();
}


void CalculatorInterface::updateGL(void)
{
    // TODO
}


GLuint CalculatorInterface::getImageTexture()
{
    return imageTexture_.getTexture();
}

