
#define GL_GLEXT_PROTOTYPES

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

#include <SFML/Window.hpp>

#define __CL_ENABLE_EXCEPTIONS

#include "filterer.h"
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


class Abs {
    // Kernel that takes a two-component (i.e. complex) image, and puts out a
    // single-component (magnitude) image

public:

    Abs() = default;
    Abs(const Abs&) = default;
    Abs(cl::Context& context, const std::vector<cl::Device>& devices);

    void operator() (cl::CommandQueue& cq, const cl::Image2D& input,
                                           const cl::Image2D& output,
                     const std::vector<cl::Event>& waitEvents
                        = std::vector<cl::Event>(),
                     cl::Event* doneEvent = nullptr);

private:

    cl::Context context_;
    cl::Kernel kernel_;

};


#define STRING(t) #t

Abs::Abs(cl::Context& context, const std::vector<cl::Device>& devices)
{
    std::string src = STRING(
        __kernel void absKernel(__read_only image2d_t input,
                          __write_only image2d_t output)
        {
            sampler_t s = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST;

            int x = get_global_id(0);
            int y = get_global_id(1);

            if (x < get_image_width(output)
                && y < get_image_height(output)) {

                float2 valIn = read_imagef(input, s, (int2)(x, y)).xy / 20.0f;
                write_imagef(output, (int2)(x, y), fast_length(valIn));
            }

        }
    );

    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(std::make_pair(src.c_str(), src.length()));

    // Compile it...
    cl::Program program(context, source);
    try {
        program.build(devices);
    } catch(cl::Error err) {
	    std::cerr 
		    << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
		    << std::endl;
	    throw;
    } 
        
    // ...and extract the useful part, viz the kernel
    kernel_ = cl::Kernel(program, "absKernel");
}



void Abs::operator() (cl::CommandQueue& cq, const cl::Image2D& input,
                                       const cl::Image2D& output,
                 const std::vector<cl::Event>& waitEvents,
                 cl::Event* doneEvent)
{
    const int wgSize = 16;

    cl::NDRange workgroupSize = {wgSize, wgSize};

    cl::NDRange globalSize = {
        roundWGs(output.getImageInfo<CL_IMAGE_WIDTH>(), wgSize), 
        roundWGs(output.getImageInfo<CL_IMAGE_HEIGHT>(), wgSize)
    }; 


    // Set all the arguments
    kernel_.setArg(0, input);
    kernel_.setArg(1, output);

    // Execute
    cq.enqueueNDRangeKernel(kernel_, cl::NullRange,
                            globalSize, workgroupSize,
                            &waitEvents, doneEvent);
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



class Main {
private:
    Dtcwt dtcwt;
    DtcwtOutput out;
    DtcwtTemps env;
    sf::Window app;

    cl::Platform platform;
    std::vector<cl::Device> devices;
    cl::Context context;
    cl::CommandQueue commandQueue; 

    cl::Image2DGL inImage;

    // For interop OpenGL/OpenCL
    GLuint texture[7];
    cl::Image2DGL dispImage[6];

	VBOBuffers buffers;
	VBOBuffers imageDisplayVertexBuffers;

    Abs abs;

    cv::VideoCapture video;

	void createTextures(int width, int height);
	void createBuffers();

public:

    Main();

    bool update();

};


void Main::createTextures(int width, int height)
{
    // Create the textures
    glGenTextures(7, texture);
    
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
    
    	// Create the associated OpenCL image
    	dispImage[n] = cl::Image2DGL(context, CL_MEM_READ_WRITE,
    								 GL_TEXTURE_2D, 0,
    								 texture[n]);
    
    }

	// Image input
	glBindTexture(GL_TEXTURE_2D, texture[6]);
    
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
    
    // Create the associated OpenCL image
    inImage = cl::Image2DGL(context, CL_MEM_READ_WRITE,
    							 GL_TEXTURE_2D, 0,
    							 texture[6]);

}


void Main::createBuffers()
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
	std::vector<float> coords = {1.f, 2.f / 3.f, 
							     0.f, 2.f / 3.f,
								 0.f, 0.f,
								 1.f, 0.f};
	glBindBuffer(GL_ARRAY_BUFFER, imageDisplayVertexBuffers.getBuffer(1));
	glBufferData(GL_ARRAY_BUFFER, coords.size()*sizeof(float), &coords[0], 
			     GL_STATIC_DRAW);

}

Main::Main()
 : app(sf::VideoMode(2*160*2, 3*120*2, 32), "SFML OpenGL"),
   video(0)
{
    try {

        app.SetActive();
        std::tie(platform, devices, context, commandQueue) = initOpenCL();

		const int width = 640, height = 480;

        const int numLevels = 6;
        const int startLevel = 1;

        // Create the DTCWT, temporaries and outputs
        dtcwt = Dtcwt(context, devices, commandQueue);
        env = dtcwt.createContext(width, height,
                                  numLevels, startLevel);
        out = DtcwtOutput(env);

        // Create the abs kernel
        abs = Abs(context, devices);

		createTextures(width, height);
		createBuffers();

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

    // Synchronise OpenGL
    glFinish();

    // Get the image from the camera

    cv::Mat picture;
    video >> picture;

    cv::Mat in = convertVideoImgToFloat(picture);
    std::cout << in.rows <<  " " << in.cols << std::endl;

    // Synchronise OpenCL
    std::vector<cl::Memory> mems(&dispImage[0], &dispImage[5] + 1);
	mems.push_back(inImage);

    commandQueue.enqueueAcquireGLObjects(&mems);

    writeImage2D(commandQueue, inImage, reinterpret_cast<float*>(in.data));

    dtcwt(commandQueue, inImage, env, out);


    for (int n = 0; n < 6; ++n)
        abs(commandQueue, out.subbands[0].sb[n], dispImage[n],
                          out.subbands[0].done);

    commandQueue.enqueueReleaseGLObjects(&mems);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
    glColor3f(1.0, 1.0, 1.0);
    glEnable(GL_TEXTURE_2D);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    for (int n = 0; n < 3; ++n) {
        for (int m = 0; m < 2; ++m) {

		    glPushMatrix();
		    glTranslatef(-1.f + m, 1.f/3.f - n * 2.f / 3.f, 0.f);

			// Select the texture
            glBindTexture(GL_TEXTURE_2D, texture[n+3*m]);

			// Select texture positioning
			glBindBuffer(GL_ARRAY_BUFFER, imageDisplayVertexBuffers.getBuffer(0));
			glTexCoordPointer(2, GL_FLOAT, 0, 0);

			// Select vertex positioning
			glBindBuffer(GL_ARRAY_BUFFER, imageDisplayVertexBuffers.getBuffer(1));
			glVertexPointer(2, GL_FLOAT, 0, 0);

			// Draw it
			glDrawArrays(GL_QUADS, 0, 4);

			glPopMatrix();

        }
    }

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
	glPointSize(2.f);

	glBindBuffer(GL_ARRAY_BUFFER, buffers.getBuffer(0));
	glVertexPointer(2, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);

	glPushMatrix();
	glDrawArrays(GL_POINTS, 0, 3);
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


