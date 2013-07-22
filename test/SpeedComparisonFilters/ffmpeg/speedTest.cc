// Copyright (C) 2013 Timothy Gale
#include "avmm/avmm.h"

#include <chrono>
#include <queue>
#include <utility>
#include <iostream>
#include <stdexcept>
#include <memory>


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

    int numFrames = 1000;


    auto prevTime = std::chrono::steady_clock::now();

    for (int n = 0; n < numFrames;) {

        // Acquire the new image
        AV::Packet packet;
        formatContext.readFrame(&packet);


        if (packet.get()->stream_index == stream) {

            AV::Frame frame;

            if (codecContext.decodeVideo(&frame, packet)) {

                ++n; 

                frame.deinterlace();

                /*std::shared_ptr<AV::Frame>
                    formattedFrame {
                        new AV::Frame {
                            codecContext.width(), 
                            codecContext.height(),
                            PIX_FMT_YUV422P
                        }
                    };

                swsContext.scale(&*formattedFrame, frame);*/

            }
        }

    }


    auto newTime = std::chrono::steady_clock::now();
    std::cout << (DurationMilliseconds(newTime - prevTime).count() / numFrames)
              << "ms\n";


    return 0;
}



