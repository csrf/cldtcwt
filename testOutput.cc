
#include <GL/glew.h>
#include <GL/glut.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

#define __CL_ENABLE_EXCEPTIONS

#include "filterer.h"
#include "clUtil.h"
#include "dtcwt.h"
#include <iomanip>

#include <ctime>

#include <stdexcept>

#include <highgui.h>

#include <GL/glxew.h>
#include <CL/cl_gl.h>


#include "cl.hpp"

std::tuple<cl::Platform, std::vector<cl::Device>, 
           cl::Context, cl::CommandQueue> 
    initOpenCL();


cl::Platform* gplatform;
std::vector<cl::Device>* gdevices;
cl::Context* gcontext;
cl::CommandQueue* gcommandQueue; 
cl::Image2D* ginImage;
Dtcwt* gdtcwt;
DtcwtOutput* gout;
DtcwtTemps* genv;
GLuint texture;


void render(void)
{
    
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glColor3f(1.0, 1.0, 1.0);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture);
    	glBegin(GL_QUADS);


        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1, 1);
		glTexCoord2f(0.0f, 1.0f); glVertex2f(-1, 1);
		glTexCoord2f(0.0f, 0.0f); glVertex2f(-1,-1);
		glTexCoord2f(1.0f, 0.0f); glVertex2f( 1,-1);
	glEnd();

	glutSwapBuffers();
    std::cout << "Called!" << std::endl;
}

void idle(void)
{
    (*gdtcwt)(*gcommandQueue, *ginImage, *genv, *gout);
    gcommandQueue->finish();
}

int main(int argc, char** argv)
{

    try {

        // Read in image
        cv::Mat bmp = cv::imread("test.bmp", 0);

        glutInit(&argc, argv);

        glutInitWindowPosition(-1, -1);
        glutInitWindowSize(bmp.cols / 2, bmp.rows / 2);
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);


        glutCreateWindow("DTCWT");

        glutDisplayFunc(render);
        glutIdleFunc(idle);


        glewInit();

        // Create the texture
        glGenTextures(1, &texture);

        glBindTexture(GL_TEXTURE_2D, texture);
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

        float img[100][100] = {0.5};
        for (int x = 0; x < 100; ++x)
            for (int y = 0; y < 100; ++y)
                img[y][x] = static_cast<float>(x) / 100.0f;
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, 100, 100, 0,
                     GL_LUMINANCE, GL_FLOAT, img);



        cl::Platform platform;
        std::vector<cl::Device> devices;
        cl::Context context;
        cl::CommandQueue commandQueue; 
        std::tie(platform, devices, context, commandQueue) = initOpenCL();

        const int numLevels = 6;
        const int startLevel = 1;


        //-----------------------------------------------------------------
        // Starting test code
  
        cl::Image2D inImage = createImage2D(context, bmp);

        std::cout << bmp.rows << " " << bmp.cols << std::endl;
        std::cout << "Creating Dtcwt" << std::endl;


        Dtcwt dtcwt(context, devices, commandQueue);

        std::cout << "Creating the DTCWT environment..." << std::endl;

        DtcwtTemps env = dtcwt.createContext(bmp.cols, bmp.rows,
                                           numLevels, startLevel);

        std::cout << "Creating the subband output images..." << std::endl;
        DtcwtOutput out(env);




        gplatform = &platform;
        gdevices = &devices;
        gcontext = &context;
        ginImage = &inImage;
        gcommandQueue = &commandQueue;
        gdtcwt = &dtcwt;
        gout = &out;
        genv = &env;



        glutMainLoop();



        std::cout << "Running DTCWT" << std::endl;



        time_t start, end;
        const int numFrames = 1000;
        time(&start);
            for (int n = 0; n < numFrames; ++n) {
                dtcwt(commandQueue, inImage, env, out);
                commandQueue.finish();
            }
        time(&end);
        std::cout << (numFrames / difftime(end, start))
		  << " fps" << std::endl;
        std::cout << numFrames << " frames in " 
                  << difftime(end, start) << "s" << std::endl;

        std::cout << "Displaying image" << std::endl;

        //for (auto& img: env.outputs[0])
        //    displayComplexImage(commandQueue, img);


    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
                     
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


