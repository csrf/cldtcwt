
#define GL_GLEXT_PROTOTYPES

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

#include <SFML/Window.hpp>

#define __CL_ENABLE_EXCEPTIONS

#include "filterer.h"
#include "abs.h"
#include "clUtil.h"
#include "dtcwt.h"
#include <iomanip>

#include <ctime>

#include <stdexcept>

#include <highgui.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <CL/cl_gl.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <GL/glext.h>
#include "findMax.h"

#include "cl.hpp"

std::tuple<cl::Platform, std::vector<cl::Device>, 
           cl::Context, cl::CommandQueue> 
    initOpenCL();





cv::Mat convertVideoImgToFloat(cv::Mat videoImg)
{
    // Convert to floating point
    cv::Mat tmp;
    videoImg.convertTo(tmp, CV_32F);

    // Now go to greyscale
    cv::Mat out;
    cv::cvtColor(tmp, out, CV_RGB2GRAY);

    return out;
}







class VBOBuffers {
public:
	
    VBOBuffers(const VBOBuffers&) = default;
	VBOBuffers(int num = 0);
	~VBOBuffers();

	GLuint getBuffer(int n);

private:

	std::vector<GLuint> buffers_;

};



VBOBuffers::VBOBuffers(int num)
 : buffers_(num)
{
	glGenBuffers(num, &buffers_[0]);
}


GLuint VBOBuffers::getBuffer(int n)
{
	return buffers_[n];
}


VBOBuffers::~VBOBuffers()
{
	glDeleteBuffers(buffers_.size(), &buffers_[0]);
}






class CLCalcs {
private:

    cl::Platform platform;
    std::vector<cl::Device> devices;
    cl::Context context;
    cl::CommandQueue commandQueue; 

    Dtcwt dtcwt;
    Abs abs;
    EnergyMap energyMap;
    FindMax findMax;

    cl::Image2D zeroImage;

    cl::Image2DGL inImage;

    DtcwtTemps env;
    DtcwtOutput out;

    cl::Image2DGL dispImage[6];

    std::vector<cl::Image2D> energyMaps;

    cl::Buffer numKps;
    cl::BufferGL keypointLocs;

public:

    CLCalcs(const CLCalcs&) = default;
    CLCalcs() = default;
    CLCalcs(int width, int height,
            GLuint textureInImage, GLuint texture[6],
            GLuint keypointLocationBuffer);


    int update();
};





CLCalcs::CLCalcs(int width, int height,
                 GLuint textureInImage, GLuint texture[6],
                 GLuint keypointLocationBuffer)
{
    std::tie(platform, devices, context, commandQueue) = initOpenCL();

    const int numLevels = 6;
    const int startLevel = 1;

    // Create the DTCWT, temporaries and outputs
    dtcwt = Dtcwt(context, devices, commandQueue);
    env = dtcwt.createContext(width, height,
                              numLevels, startLevel);
    out = DtcwtOutput(env);

    // Create energy maps for each output level (other than the last,
    // which is only there for coarse detections)
    for (int l = 0; l < (out.subbands.size() - 1); ++l) {
        energyMaps.push_back(
            createImage2D(context, 
                out.subbands[l].sb[0].getImageInfo<CL_IMAGE_WIDTH>(),
                out.subbands[l].sb[0].getImageInfo<CL_IMAGE_HEIGHT>())
        );
    }

    // Zero image for use off the ends of maximum finding
    float zerof = 0;
    zeroImage = {
        context, 
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), 
        1, 1, 0,
        &zerof
    };

    // Create the kernels
    abs = Abs(context, devices);
    energyMap = EnergyMap(context, devices);
    findMax = FindMax(context, devices);

    std::vector<float> zeroV = {0.f};
    numKps = createBuffer(context, commandQueue, zeroV);
 
    // Create the associated OpenCL image
    inImage = cl::Image2DGL(context, CL_MEM_READ_WRITE,
    					    GL_TEXTURE_2D, 0,
    					    textureInImage);

    for (int n = 0; n < 6; ++n) {

    	// Create the associated OpenCL image
    	dispImage[n] = cl::Image2DGL(context, CL_MEM_READ_WRITE,
    								 GL_TEXTURE_2D, 0,
    								 texture[n]);

    }
 
    keypointLocs = cl::BufferGL(context, CL_MEM_READ_WRITE,
                                keypointLocationBuffer);
}



int CLCalcs::update()
{
    // Synchronise OpenCL
    std::vector<cl::Memory> mems(&dispImage[0], &dispImage[5] + 1);
    mems.push_back(keypointLocs);
    mems.push_back(inImage);

    commandQueue.enqueueAcquireGLObjects(&mems);

    dtcwt(commandQueue, inImage, env, out);

    // Calculate energy maps
    for (int l = 0; l < energyMaps.size(); ++l)
        energyMap(commandQueue, out.subbands[l], energyMaps[l]);

    // Look for peaks in them
    for (int l = 0; l < energyMaps.size(); ++l)
        ;

    writeBuffer(commandQueue, numKps, std::vector<int> {0});
    findMax(commandQueue, energyMaps[0], zeroImage, energyMaps[1], 0.1f,
            keypointLocs, numKps);

    int numOutputsVal;
    commandQueue.enqueueReadBuffer(numKps, CL_TRUE, 0, sizeof(int),
                                   &numOutputsVal);

    std::cout << numOutputsVal << std::endl;

    for (int n = 0; n < 6; ++n)
        abs(commandQueue, out.subbands[0].sb[n], dispImage[n],
                          out.subbands[0].done);

    commandQueue.enqueueReleaseGLObjects(&mems);
    commandQueue.finish();

    return numOutputsVal;
}



class Main {
private:

    sf::Window app;

    cv::VideoCapture video;

    // For interop OpenGL/OpenCL
    GLuint textureInImage;
    GLuint texture[6];

	VBOBuffers buffers;
	VBOBuffers imageDisplayVertexBuffers;
    VBOBuffers keypointLocationBuffers;

    CLCalcs clCalcs;

	void createTextures(int width, int height);
	void createBuffers(int numKeypointLocationBuffers);

public:

    Main();

    bool update();

};


void Main::createTextures(int width, int height)
{
    // Create the textures
    glGenTextures(6, texture);
    glGenTextures(1, &textureInImage);
    
	// Subband outputs
    for (int n = 0; n < 6; ++n) {
    
    	glBindTexture(GL_TEXTURE_2D, texture[n]);
    
    	// Set up the texture display properties
    	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    
    	// Put it to the right size, filling with zeros
    	std::vector<float> zeros(width * height, 0.0f);
    	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, 
    				 width / 4, height / 4, 0,
    				 GL_LUMINANCE, GL_FLOAT, &zeros[0]);
    
   
    }

	// Image input
	glBindTexture(GL_TEXTURE_2D, textureInImage);
    
    // Set up the texture display properties
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    
    // Put it to the right size, filling with zeros
    std::vector<float> zeros(width * height, 0.0f);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, 
    			 width, height, 0,
    			 GL_LUMINANCE, GL_FLOAT, &zeros[0]);
}


void Main::createBuffers(int numKeypointLocationBuffers)
{
	buffers = VBOBuffers(2);


	float pcoords[3*2] = {0.f, 0.5f, 0.5f, 0.5f, -1.0f, 0.0f};
	glBindBuffer(GL_ARRAY_BUFFER, buffers.getBuffer(0));
	glBufferData(GL_ARRAY_BUFFER, 6*sizeof(float), pcoords, GL_STATIC_DRAW);

	float colours[4*3] = {1.f, 1.f, 1.f, 1.f,
					      1.f, 1.f, 1.f, 1.f,
						  1.f, 1.f, 1.f, 1.f};
	glBindBuffer(GL_ARRAY_BUFFER, buffers.getBuffer(1));
	glBufferData(GL_ARRAY_BUFFER, 12*sizeof(float), colours, GL_STATIC_DRAW);

	// The buffers setting coords for displaying the images: first, the texture
	// coordinates, then the vertex coordinates
	imageDisplayVertexBuffers = VBOBuffers(2);

	// Texture coordinates
	std::vector<float> texCoords = {1.f, 0.f, 
								    0.f, 0.f,
									0.f, 1.f,
									1.f, 1.f};
	glBindBuffer(GL_ARRAY_BUFFER, imageDisplayVertexBuffers.getBuffer(0));
	glBufferData(GL_ARRAY_BUFFER, texCoords.size()*sizeof(float), &texCoords[0], 
			     GL_STATIC_DRAW);
	

	// Coordinates of the vertices
	std::vector<float> coords = {0.5f, 2.f / 3.f, 
							     0.0f, 2.f / 3.f,
								 0.0f, 0.f,
								 0.5f, 0.f};
	glBindBuffer(GL_ARRAY_BUFFER, imageDisplayVertexBuffers.getBuffer(1));
	glBufferData(GL_ARRAY_BUFFER, coords.size()*sizeof(float), &coords[0], 
			     GL_STATIC_DRAW);

    const int maxNumKeypoints = 1000;

    // For the keypoint location extraction
    std::vector<float> kps(maxNumKeypoints * 2);
    keypointLocationBuffers = VBOBuffers(numKeypointLocationBuffers);
	glBindBuffer(GL_ARRAY_BUFFER, keypointLocationBuffers.getBuffer(0));
	glBufferData(GL_ARRAY_BUFFER, kps.size()*sizeof(float), &kps[0], 
			     GL_STATIC_DRAW);

}

Main::Main()
 : app(sf::VideoMode(640*2, 3*120*2, 32), "SFML OpenGL"),
   video(0)
{
    try {

        app.SetActive();

		const int width = 640, height = 480;

		createTextures(width, height);
		createBuffers(1);

        clCalcs = CLCalcs(width, height,
                          textureInImage, texture,
                          keypointLocationBuffers.getBuffer(0));

    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
 
}








bool Main::update(void)
{
    if (!app.IsOpened())
        return false;

    sf::Event event;
    while (app.GetEvent(event)) {

        if (event.Type == sf::Event::Closed)
            return false;

        if (event.Type == sf::Event::Resized) {
            std::cout << "Resized" << std::endl;
        }
    }

    app.SetActive();



    // Get the image from the camera

    cv::Mat picture;
    video >> picture;

    cv::Mat in = convertVideoImgToFloat(picture);
    std::cout << in.rows <<  " " << in.cols << std::endl;

    in /= 256.f;
    // Copy matrix contents to the inImage/textureInImage with OpenGL (since
    // OpenCL seems to have issues doing enqueueWriteImage to a shared
    // texture)
    glBindTexture(GL_TEXTURE_2D, textureInImage);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, in.cols, in.rows,
                    GL_LUMINANCE, GL_FLOAT, in.data); 
                    

    // Synchronise OpenGL
    glFinish();

    int numKeypoints = clCalcs.update();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
    glColor3f(1.0, 1.0, 1.0);
    glEnable(GL_TEXTURE_2D);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    for (int n = 0; n < 3; ++n) {
        for (int m = 0; m < 2; ++m) {

		    glPushMatrix();
		    glTranslatef(m / 2.f, 1.f/3.f - n * 2.f / 3.f, 0.f);

            int sbIdx = (m == 0) ? n : (5 - n);
			// Select the texture
            glBindTexture(GL_TEXTURE_2D, texture[sbIdx]);

			// Select texture positioning
			glBindBuffer(GL_ARRAY_BUFFER,
                         imageDisplayVertexBuffers.getBuffer(0));
			glTexCoordPointer(2, GL_FLOAT, 0, 0);

			// Select vertex positioning
			glBindBuffer(GL_ARRAY_BUFFER, 
                         imageDisplayVertexBuffers.getBuffer(1));
			glVertexPointer(2, GL_FLOAT, 0, 0);

			// Draw it
			glDrawArrays(GL_QUADS, 0, 4);

			glPopMatrix();

        }
    }

    // Display the original image

    glPushMatrix();

    glTranslatef(-1.f, -1.f/3.f, 0.f);
    glScalef(2.f, 2.f, 1.f);
    // Select the texture
    glBindTexture(GL_TEXTURE_2D, textureInImage);

    // Select texture positioning
    glBindBuffer(GL_ARRAY_BUFFER, imageDisplayVertexBuffers.getBuffer(0));
    glTexCoordPointer(2, GL_FLOAT, 0, 0);

    // Select vertex positioning
    glBindBuffer(GL_ARRAY_BUFFER, imageDisplayVertexBuffers.getBuffer(1));
    glVertexPointer(2, GL_FLOAT, 0, 0);

    // Draw it
    glDrawArrays(GL_QUADS, 0, 4);


	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);

	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);

		

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	glColor4f(1.0, 0.0, 0.0, 1.0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_POINT_SMOOTH);
	glPointSize(7.f);

	glBindBuffer(GL_ARRAY_BUFFER, keypointLocationBuffers.getBuffer(0));
	glVertexPointer(2, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);

    glTranslatef(0.f, 2.f/3.f, 0.f);
    glScalef(1.f / 160.f * 0.5f, -1.f / 120.f * 2.f / 3.f, 1.f);
   
	glDrawArrays(GL_POINTS, 0, numKeypoints);
	glPopMatrix();

	glDisableClientState(GL_VERTEX_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    app.Display();

    return true;
}


int main(int argc, char** argv)
{
    Main mainObj;
                
    while (mainObj.update())
        ;

    return 0;
}


std::tuple<cl::Platform, std::vector<cl::Device>, 
           cl::Context, cl::CommandQueue> 
initOpenCL()
{
    // Get platform, devices, command queue

    // Retrive platform information
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.size() == 0)
        throw std::runtime_error("No platforms!");

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);

    cl_context_properties props[] = { 
        CL_GLX_DISPLAY_KHR, (intptr_t) glXGetCurrentDisplay(),
        CL_GL_CONTEXT_KHR, (intptr_t) glXGetCurrentContext(),
        0
    };
    // Create a context to work in 
    cl::Context context(devices, props);

    // Ready the command queue on the first device to hand
    cl::CommandQueue commandQueue(context, devices[0]);

    return std::make_tuple(platforms[0], devices, context, commandQueue);
}


