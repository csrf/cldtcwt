

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

#include <CL/cl_gl.h>

#include <GL/glx.h>

#include "cl.hpp"

std::tuple<cl::Platform, std::vector<cl::Device>, 
           cl::Context, cl::CommandQueue> 
    initOpenCL();


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

                float2 valIn = read_imagef(input, s, (int2)(x, y)).xy;
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






class Main {
private:
    Dtcwt dtcwt;
    DtcwtOutput out;
    DtcwtTemps env;
    GLuint texture;
    sf::Window app;

    cl::Platform platform;
    std::vector<cl::Device> devices;
    cl::Context context;
    cl::CommandQueue commandQueue; 

    cl::Image2D inImage;

    // For interop OpenGL/OpenCL
    cl::Image2DGL dispImage;

    Abs abs;

public:

    Main();

    bool update();

};


Main::Main()
 : app(sf::VideoMode(800, 600, 32), "SFML OpenGL")
{
    try {


        // Read in image
        cv::Mat bmp = cv::imread("test.bmp", 0);

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

        app.SetActive();


        std::tie(platform, devices, context, commandQueue) = initOpenCL();

        abs = Abs(context, devices);

        dispImage = cl::Image2DGL(context, CL_MEM_READ_WRITE,
                                  GL_TEXTURE_2D, 0,
                                  texture);


        const int numLevels = 6;
        const int startLevel = 1;


        //-----------------------------------------------------------------
        // Starting test code
  
        cl::Image2D inImage = createImage2D(context, bmp);

        std::cout << bmp.rows << " " << bmp.cols << std::endl;
        std::cout << "Creating Dtcwt" << std::endl;


        dtcwt = Dtcwt(context, devices, commandQueue);

        std::cout << "Creating the DTCWT environment..." << std::endl;

        env = dtcwt.createContext(bmp.cols, bmp.rows,
                                  numLevels, startLevel);

        std::cout << "Creating the subband output images..." << std::endl;
        out = DtcwtOutput(env);

        dtcwt(commandQueue, inImage, env, out);


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

    // Synchronise OpenCL
    std::vector<cl::Memory> mems = {dispImage};
    commandQueue.enqueueAcquireGLObjects(&mems);

    abs(commandQueue, out.subbands[0].sb[0], dispImage);

    commandQueue.enqueueReleaseGLObjects(&mems);

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


