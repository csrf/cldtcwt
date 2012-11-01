#include "DisplayOutput/viewer.h"
#include <vector>


Viewer::Viewer(int width, int height)
 : window(sf::VideoMode(width*2, height*2, 32), "SFML OpenGL"),
   imageDisplayVertexBuffers_(2)
{
	// The buffers setting coords for displaying the images: first, the texture
	// coordinates, then the vertex coordinates

	// Texture coordinates
	glBindBuffer(GL_ARRAY_BUFFER, imageDisplayVertexBuffers_.getBuffer(0));

	std::vector<float> texCoords = {1.f, 0.f, 
								    0.f, 0.f,
									0.f, 1.f,
									1.f, 1.f};

	glBufferData(GL_ARRAY_BUFFER, texCoords.size()*sizeof(float), 
                 &texCoords[0], 
			     GL_STATIC_DRAW);
	

	// Coordinates of the vertices
	glBindBuffer(GL_ARRAY_BUFFER, imageDisplayVertexBuffers_.getBuffer(1));

	std::vector<float> coords = {0.5f, 0.5f, 
							     0.0f, 0.5f,
								 0.0f, 0.f,
								 0.5f, 0.f};

	glBufferData(GL_ARRAY_BUFFER, coords.size()*sizeof(float), &coords[0], 
			     GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}



void Viewer::setImageTexture(GLuint texture)
{
    imageTexture_ = texture;
}


#include <iostream>

void Viewer::update()
{
    if (!window.IsOpened())
        return;

    sf::Event event;
    while (window.GetEvent(event)) {

        // If the user tried to close the window, flag that everything is
        // done
        if (event.Type == sf::Event::Closed) {
            done_ = true;
            return;
        }

    }

    glFinish();
    window.SetActive();

    // Display the window

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_TEXTURE_2D);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glEnableClientState(GL_VERTEX_ARRAY);

	// Select texture positioning
    glBindBuffer(GL_ARRAY_BUFFER, imageDisplayVertexBuffers_.getBuffer(0));
    glTexCoordPointer(2, GL_FLOAT, 0, 0);

    // Select vertex positioning
    glBindBuffer(GL_ARRAY_BUFFER, imageDisplayVertexBuffers_.getBuffer(1));
    glVertexPointer(2, GL_FLOAT, 0, 0);

    // Select the texture
    glBindTexture(GL_TEXTURE_2D, imageTexture_);

    glColor4f(1.0, 1.0, 1.0, 1.0);

    // Display the original image

    glPushMatrix();

    //glTranslatef(-1.f, -1.f, 0.f);
    //glScalef(1.f, 1.f, 1.f);

    // Draw it
    glDrawArrays(GL_QUADS, 0, 4);

    glPopMatrix();

	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);

	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);

    window.Display();

}


bool Viewer::isDone()
{
    return done_;
}


