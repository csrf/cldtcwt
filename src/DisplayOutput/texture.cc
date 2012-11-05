#include "DisplayOutput/texture.h"
#include <vector>


GLTexture::GLTexture() 
{
    // Create the object
    glGenTextures(1, &texture_);

    // Select it
    glBindTexture(GL_TEXTURE_2D, texture_);

    // Set up the texture display properties
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
}



GLTexture::GLTexture(GLint internalFormat, GLsizei width, GLsizei height)
    : GLTexture()
{
    setProperties(internalFormat, width, height);
}


GLTexture::GLTexture(GLTexture&& obj)
{
    // Acquire
    texture_ = obj.texture_;

    // Deactivate the other texture
    obj.texture_ = 0;
}



GLTexture::~GLTexture()
{
    if (texture_ != 0)
        glDeleteTextures(1, &texture_);
}



GLTexture& GLTexture::operator= (GLTexture&& obj)
{
    // Get rid of current texture (if any)
    if (texture_ != 0)
        glDeleteTextures(1, &texture_);

    // Acquire
    texture_ = obj.texture_;

    // Deactivate the other texture
    obj.texture_ = 0;
   
    return *this;
}



GLuint GLTexture::getTexture()
{
    return texture_;
}


void GLTexture::setProperties(GLint internalFormat, 
                              GLsizei width, GLsizei height)
{
    // Select it
    glBindTexture(GL_TEXTURE_2D, texture_);

    // Create zeros of the right size
    std::vector<char> zeros(width * height, 0);

    // Initialise to OpenGL format, size
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat,
                 width, height, 0, 
                 GL_RED, GL_BYTE, &zeros[0]);
}


