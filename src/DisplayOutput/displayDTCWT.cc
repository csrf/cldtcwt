#include <iostream>
#include <tuple>
#include <stdexcept>

#include <SFML/Window.hpp>

#define GL_GLEXT_PROTOTYPES

#include <CL/cl_gl.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <GL/glext.h>
#include "KeypointDetector/findMax.h"

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include <highgui.h>

#include "DisplayOutput/viewer.h"
#include "DisplayOutput/calculatorInterface.h"


#include "VideoReader.h"

// For timing
#include <sys/timeb.h>


#include "DisplayOutput/VideoReader.h"

std::tuple<cl::Platform, std::vector<cl::Device>, cl::Context> 
    initOpenCL();

int main(void)
{
    Viewer viewer(1280, 720);

    VideoReader videoReader("/dev/video0", 1280, 720);
    videoReader.startCapture();

    //cv::VideoCapture video(0);

    //video.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    //video.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

    cl::Platform platform;
    cl::Context context;
    std::vector<cl::Device> devices;
    std::tie(platform, devices, context) = initOpenCL();
   
    CalculatorInterface ci(context, devices[0], 1280, 720);   


    int n = 0;
    while (1) {

        timeb start, end;

        //cv::Mat input;
        //video >> input;

        VideoReaderBuffer buffer = videoReader.getFrame();
        ftime(&start);
        ci.processImage(buffer.start, buffer.length);
        videoReader.returnBuffers();

        viewer.setImageTexture(ci.getImageTexture());
        viewer.update();

        ftime(&end);

        // Work out what the difference between these is
        double t = end.time - start.time 
                 + 0.001 * (end.millitm - start.millitm);
        
        std::cout << n++ << " " << (1000*t) << "ms\n";

        if (viewer.isDone())
            break;
    }

    videoReader.stopCapture();

    return 0;
}



std::tuple<cl::Platform, std::vector<cl::Device>, cl::Context> 
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

    return std::make_tuple(platforms[0], devices, context);
}


