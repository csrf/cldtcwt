#ifndef VIEWER_H
#define VIEWER_H

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include <array>

#include <SFML/Window.hpp>

#include "DisplayOutput/VBOBuffer.h"


class Viewer {

    static const int numSubbands = 6;

private:

    float width_, height_;

    // The actual display window
    sf::Window window;

    // Name of the OpenGL texture
    GLuint imageTexture_ = 0;

    // Name of the energy map texture
    GLuint energyMapTexture_ = 0;

    // And those of the subbands
    std::array<GLuint, numSubbands> subbandTextures2_;
    std::array<GLuint, numSubbands> subbandTextures3_;

    // Two buffers, used for placing (0) texture coordinates
    // and (1) vertex coordinates, both in 2D
    VBOBuffers imageDisplayVertexBuffers_;

    // Controls drawing the keypoints
    GLuint keypointLocations_ = 0;
    size_t numKeypointLocations_ = 0;
    size_t numFloatsPerKeypoint_ = 0;

    // Whether the user has asked for the window to be closed
    bool done_ = false;

    void drawPicture();
    void drawEnergyMap();
    void drawSubbands(const GLuint textures[]);

public:

    Viewer(int width, int height);

    // Call after creation and before any drawing.  Needed because AMD
    // drivers don't like transfering buffers to OpenCL if we've
    // already played around with buffers before initialising OpenCL.
    void initBuffers();


    void setImageTexture(GLuint texture);
    void setEnergyMapTexture(GLuint texture);
    void setSubband2Texture(int subband, GLuint texture);
    void setSubband3Texture(int subband, GLuint texture);
    
    void setKeypointLocations(GLuint buffer, size_t numKeypoints);
    void setNumFloatsPerKeypoint(size_t);

    void update();
    // Update the display

    bool isDone();
    // Whether the window has been closed

};

#endif


