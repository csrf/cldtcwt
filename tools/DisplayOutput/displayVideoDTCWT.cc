// Copyright (C) 2013 Timothy Gale
#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>

#include "DisplayOutput/calculatorInterface.h"
#include "DisplayOutput/viewer.h"

#include <GL/glx.h>

#include "avmm/avmm.h"

#include "hdf5/hdfwriter.h"

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


void writeResults(cl::CommandQueue& cq, Calculator& cal, int numKeypoints,
                  HDFWriter& output)
{
    // Read the locations
    std::vector<cl_float> locations(4*numKeypoints);
    cq.enqueueReadBuffer(cal.keypointLocations(), CL_FALSE, 0, 
                         sizeof(cl_float) * locations.size(), 
                         &locations[0]);

    // Read the descriptors
    std::vector<cl_float> descriptors(2*6*14*numKeypoints);
    cq.enqueueReadBuffer(cal.keypointDescriptors(), CL_FALSE, 0, 
                         sizeof(cl_float) * descriptors.size(),
                         &descriptors[0]);

    // Write them all out, when ready
    cq.finish();

    output.append(numKeypoints, &locations[0], &descriptors[0]);
}


int main(int argc, char* argv[])
{
    AV::registerAll();

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " VideoFilename [OutputFilename.h5]"
                  << std::endl;

        return -1;
    }

    bool writeOutput = argc == 3;

    // Initialise the video reader
    AV::FormatContext formatContext {argv[1]};
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

    cl::CommandQueue cq {context, devices[0]};
   
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

    HDFWriter fileOutput;
    
    if (writeOutput) 
        fileOutput = HDFWriter(argv[2], 2*6*14);


    auto prevTime = std::chrono::steady_clock::now();
    int n = 0;

    bool eof = false;

    while (1) {


        if (!eof && !ready.empty()) {
            
            // Acquire the new image

            AV::Packet packet;

            // Check for end of file
            if (formatContext.readFrame(&packet)) {
                eof = true;
                continue;
            }


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

                // Write to file
                if (writeOutput)
                    writeResults(cq, ci->getCalculator(), numKPs, fileOutput);

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

        if (processing.empty() && eof)
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


