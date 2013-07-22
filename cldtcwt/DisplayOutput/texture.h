// Copyright (C) 2013 Timothy Gale
#ifndef TEXTURE_H
#define TEXTURE_H

#include <GL/gl.h>


class GLTexture {
// Wrapper for a texture, that hides some of the details of setting up
// display properties and setting it to the right size.  Having it the
// right size already can be rather useful when it needs to 

private:
    GLuint texture_ = 0;
    // By default, use the null value that OpenGL should never use.  This
    // means we know whether it's been deleted already or not.

public:

    GLTexture();
    // Create without specifying the texture size/format

    GLTexture(GLint internalFormat, GLsizei width, GLsizei height);
    // Create with the given properties.  Useful for setting to the
    // correct size/format for OpenCL output.

    GLTexture(GLTexture&&);
    // Move another texture in, acquiring ownership of it
    
    GLTexture& operator= (GLTexture&&);
    // Likewise, move another texture in, but this time also get rid of the
    // current one.

    ~GLTexture();

    void setProperties(GLint internalFormat, 
                       GLsizei width, GLsizei height);
    // See glTexImage2D for the possibly property values

    GLuint getTexture();
    // Get the OpenGL texture name
};


#endif

