#include "DisplayOutput/viewer.h"
#include <vector>


Viewer::Viewer(int width, int height)
 : width_(width), height_(height),
   window(sf::VideoMode(width*1.5, height*1.5, 32), "SFML OpenGL"),
   imageDisplayVertexBuffers_(2)
{
}



void Viewer::initBuffers()
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

	std::vector<float> coords = {1.0f, 1.0f, 
							     0.0f, 1.0f,
								 0.0f, 0.0f,
								 1.0f, 0.0f};

	glBufferData(GL_ARRAY_BUFFER, coords.size()*sizeof(float), &coords[0], 
			     GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

}


void Viewer::setImageTexture(GLuint texture)
{
    imageTexture_ = texture;
}



void Viewer::setEnergyMapTexture(GLuint texture)
{
    energyMapTexture_ = texture;
}



void Viewer::setSubband2Texture(int subband, GLuint texture)
{
    subbandTextures2_[subband] = texture;
}



void Viewer::setSubband3Texture(int subband, GLuint texture)
{
    subbandTextures3_[subband] = texture;
}

void Viewer::setKeypointLocations(GLuint buffer, size_t numKeypoints)
{
    keypointLocations_ = buffer;
    numKeypointLocations_ = numKeypoints;
}

void Viewer::setNumFloatsPerKeypoint(size_t n)
{
    numFloatsPerKeypoint_ = n;
}

#include <iostream>
#include <sys/timeb.h>

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

    window.SetActive();

    // Set up the area of the rendering region
    glViewport(0, 0, window.GetWidth(), window.GetHeight());
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1, 0.5, -0.5, 1, 0, 2);


    // Display the window

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_TEXTURE_2D);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glEnableClientState(GL_VERTEX_ARRAY);

	// Select texture positioning
    glBindBuffer(GL_ARRAY_BUFFER, imageDisplayVertexBuffers_.getBuffer(0));
    glTexCoordPointer(2, GL_FLOAT, 0, 0);

    drawPicture();

    drawEnergyMap();

    // Draw the level 2 subbands
    glPushMatrix();
    glTranslatef(0.f, 0.25f, 0.f);
    glScalef(0.25f, 0.25f, 0.f);
    drawSubbands(&subbandTextures2_[0]);
    glPopMatrix();

    // Draw the level 3 subbands
    glPushMatrix();
    glTranslatef(-1.f, -0.5f, 0.f);
    glScalef(0.125f, 0.125f, 0.f);
    drawSubbands(&subbandTextures3_[0]);
    glPopMatrix();


	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);

	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);

    timeb t1, t2;
    window.Display();
    ftime(&t1);
    glFlush();
    ftime(&t2);

                double dt = t2.time - t1.time 
                         + 0.001 * (t2.millitm - t1.millitm);
    std::cout << "R " << (1000*dt) << "ms\n";
}


void Viewer::drawPicture() 
{
    // Draw the webcam picture

    // Select the texture
    glBindTexture(GL_TEXTURE_2D, imageTexture_);

    // Display the original image

    glPushMatrix();

    glTranslatef(-1.f, 0.f, 0.f);

    // Select vertex positioning
    glBindBuffer(GL_ARRAY_BUFFER, imageDisplayVertexBuffers_.getBuffer(1));
    glVertexPointer(2, GL_FLOAT, 0, 0);


    // Draw it
    glDrawArrays(GL_QUADS, 0, 4);

    // Draw the keypoints

	// Draw with red, 7-pixel large dots
	glColor4f(0.0, 0.7, 0.0, 1.0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_POINT_SMOOTH);
	glPointSize(5.f);

	glBindBuffer(GL_ARRAY_BUFFER, keypointLocations_);
	glVertexPointer(2, GL_FLOAT, numFloatsPerKeypoint_ * sizeof(float), 0);

    // Got to the right place to overlay the display: the middle of the
    // image 
    glTranslatef(0.5f, 0.5f, 0.f);
    glScalef(1.f / width_, -1.f / height_, 1.f);
   
    glDisable(GL_TEXTURE_2D);

	glDrawArrays(GL_POINTS, 0, numKeypointLocations_);

    glEnable(GL_TEXTURE_2D);

	glColor4f(1.0, 1.0, 1.0, 1.0);
    glPopMatrix();
}



void Viewer::drawEnergyMap() 
{
    // Draw the energy map picture

    // Select the texture
    glBindTexture(GL_TEXTURE_2D, energyMapTexture_);

    // Display the original image

    glPushMatrix();

    glTranslatef(0.f, -0.25f, 0.f);
    glScalef(0.25f, 0.25f, 0.f);

    // Select vertex positioning
    glBindBuffer(GL_ARRAY_BUFFER, imageDisplayVertexBuffers_.getBuffer(1));
    glVertexPointer(2, GL_FLOAT, 0, 0);


    // Draw it
    glDrawArrays(GL_QUADS, 0, 4);

    glPopMatrix();
}



void Viewer::drawSubbands(const GLuint textures[]) 
{
    // Coordinates to display at
    std::vector<std::array<int, 2>> positions = {
        {0, 0}, {1, 0}, {2, 0},
        {2, 1}, {1, 1}, {0, 1}
    };

    // Select vertex positioning
    glBindBuffer(GL_ARRAY_BUFFER, imageDisplayVertexBuffers_.getBuffer(1));
    glVertexPointer(2, GL_FLOAT, 0, 0);

    for (int n = 0; n < positions.size(); ++n) {
        // Select the texture
        glBindTexture(GL_TEXTURE_2D, textures[n]);

        glPushMatrix();

        glTranslatef(positions[n][1], positions[n][0], 0.f);

        // Draw it
        glDrawArrays(GL_QUADS, 0, 4);

        glPopMatrix();
    }
}



bool Viewer::isDone()
{
    return done_;
}


