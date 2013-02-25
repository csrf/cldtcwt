#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>

#include "DisplayOutput/calculatorInterface.h"
#include "DisplayOutput/viewer.h"

#include <GL/glx.h>

#include "VideoReader.h"

#include <chrono>
#include <queue>
#include <utility>
#include <iostream>
#include <tuple>
#include <stdexcept>


#include "DisplayOutput/VideoReader.h"

std::tuple<cl::Platform, std::vector<cl::Device>, cl::Context> 
    initOpenCL();

typedef std::chrono::duration<double, std::milli>
    DurationMilliseconds;

int main(void)
{
    //const size_t width = 1280, height = 720;
    const size_t width = 640, height = 480;
    Viewer viewer(width, height);


    cl::Platform platform;
    cl::Context context;
    std::vector<cl::Device> devices;
    std::tie(platform, devices, context) = initOpenCL();
   
    viewer.initBuffers();

    CalculatorInterface ci1(context, devices[0], width, height);   
    CalculatorInterface ci2(context, devices[0], width, height);   
    CalculatorInterface ci3(context, devices[0], width, height);   

    std::queue<CalculatorInterface*> ready;
    std::queue<std::pair<CalculatorInterface*, VideoReaderBuffer>> processing;
    ready.push(&ci1);
    ready.push(&ci2);
    ready.push(&ci3);

    // Set up the keypoint transfer format
    viewer.setNumFloatsPerKeypoint(ci1.getNumFloatsPerKeypointLocation());

    VideoReader videoReader("/dev/video0", width, height);
    videoReader.startCapture();

    auto prevTime = std::chrono::steady_clock::now();
    int n = 0;

    while (1) {


        if (!ready.empty()) {
            
            // Acquire the new image
            VideoReaderBuffer buffer = videoReader.getFrame();

            // Set it being processed
            ready.front()->processImage(buffer.start, buffer.length);

            // Transfer the calculator to the processing queue
            processing.push(std::make_pair(ready.front(), buffer));
            ready.pop();

        }


        if (!processing.empty()) {

            CalculatorInterface* ci = processing.front().first;

            if (ci->isDone()) {

                ci->waitUntilDone();
     
                // Return the buffer
                videoReader.returnBuffer(processing.front().second);

                // Set the texture sources for the viewer
                viewer.setImageTexture(ci->getImageTexture());
                for (int n = 0; n < 6; ++n) {
                    viewer.setSubband2Texture(n, ci->getSubband2Texture(n));
                    viewer.setSubband3Texture(n, ci->getSubband3Texture(n));
                }
                size_t numKPs = ci->getNumKeypointLocations();
                viewer.setEnergyMapTexture(ci->getEnergyMapTexture());
                viewer.setKeypointLocations(ci->getKeypointLocations(),
                                            numKPs);

                viewer.update();

                // Transfer to the ready queue
                ready.push(ci);
                processing.pop();
                
                auto newTime = std::chrono::steady_clock::now();

                // Work out what the difference between these is

                
                std::cout << n++ << " " << numKPs 
                                 << " " << 
                        DurationMilliseconds(newTime - prevTime).count()
                                 << "ms\n";

                prevTime = newTime;

            } 
        }

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


