#ifndef VIEWER_H
#define VIEWER_H

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include <SFML/Window.hpp>

#include "DisplayOutput/VBOBuffer.h"


class Viewer {

private:
    // The actual display window
    sf::Window window;

    // Name of the OpenGL texture
    GLuint imageTexture_ = 0;

    // Two buffers, used for placing (0) texture coordinates
    // and (1) vertex coordinates, both in 2D
    VBOBuffers imageDisplayVertexBuffers_;

    // Whether the user has asked for the window to be closed
    bool done_ = false;


public:

    Viewer(int width, int height);

    void setImageTexture(GLuint texture);


    void update();
    // Update the display

    bool isDone();
    // Whether the window has been closed

};

#endif


