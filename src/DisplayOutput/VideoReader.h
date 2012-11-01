#ifndef VIDEOREADER_H
#define VIDEOREADER_H

#include <libv4l2.h>

class VideoReader {
    // Class for reading from the webcam.  The idea is that it should be able to write
    // into a user buffer.  If that user buffer happens to be memory mapped into a PBO
    // of the graphics card, so much the better...

private:

    int fd_ = -1;
    // File descriptor; by default, invalid

    int numBuffers_ = 0;
    // Number of buffers the streaming has allocated



public:

    // Move resources from another
    VideoReader(VideoReader&&);

    VideoReader(const char* filename, int width, int height);
    ~VideoReader();

    void startCapture();
    // Start streaming in
};

#endif

