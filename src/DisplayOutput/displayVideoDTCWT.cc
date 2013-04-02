#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>

#include "calculatorInterface.h"
#include "viewer.h"

#include <GL/glx.h>

#include "avmm/avmm.h"

#include <chrono>
#include <queue>
#include <utility>
#include <iostream>
#include <tuple>
#include <stdexcept>
#include <memory>


std::tuple<cl::Platform, std::vector<cl::Device>, cl::Context> 
    initOpenCL();

typedef std::chrono::duration<double, std::milli>
    DurationMilliseconds;

int main(void)
{
    AV::registerAll();

    // Initialise the video reader
    AV::FormatContext formatContext
        {"/home/teg28/MATLAB/AnalyseVideoTracker/video/PVTRA102a10.mov"};
    formatContext.findStreamInfo();

    // Get which stream to read and the codec
    AVCodec* codec;
    int stream = formatContext.findBestStream(AVMEDIA_TYPE_VIDEO, 
                                              -1, -1, 
                                              &codec);

    // Open a decoding context with the codec
    AV::CodecContext codecContext 
        {formatContext.getStreamCodecContext(stream)};
    codecContext.open(codec);

    SWS::Context swsContext {
        codecContext.width(), codecContext.height(),
        codecContext.pixelFormat(),
        codecContext.width(), codecContext.height(),
        PIX_FMT_YUV422P,
        SWS_POINT
    };


    const size_t width = codecContext.width(), 
                 height = codecContext.height();
    //const size_t width = 640, height = 480;
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
    std::queue<std::pair<CalculatorInterface*, std::shared_ptr<AV::Frame>>> 
        processing;
    ready.push(&ci1);
    ready.push(&ci2);
    ready.push(&ci3);

    // Set up the keypoint transfer format
    viewer.setNumFloatsPerKeypoint(ci1.getNumFloatsPerKeypointLocation());






    auto prevTime = std::chrono::steady_clock::now();
    int n = 0;

    while (1) {


        if (!ready.empty()) {
            
            // Acquire the new image

            AV::Packet packet;
            formatContext.readFrame(&packet);


            if (packet.get()->stream_index == stream) {

                AV::Frame frame;

                if (codecContext.decodeVideo(&frame, packet)) {

                    frame.deinterlace();

                    std::shared_ptr<AV::Frame>
                        formattedFrame {
                            new AV::Frame {
                                codecContext.width(), 
                                codecContext.height(),
                                PIX_FMT_YUV422P
                            }
                        };

                    swsContext.scale(&*formattedFrame, frame);

                    // Set it being processed
                    ready.front()->processImage(formattedFrame->getData(),
                                            formattedFrame->getWidth() 
                                             * formattedFrame->getHeight());

                    // Transfer the calculator to the processing queue
                    processing.push(std::make_pair(ready.front(), 
                                                   formattedFrame));
                    ready.pop();

                }
            }

        }


        if (!processing.empty()) {

            CalculatorInterface* ci = processing.front().first;

            if (ci->isDone()) {

                ci->waitUntilDone();
     
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


