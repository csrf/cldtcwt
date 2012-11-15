#ifndef VIDEOREADER_H
#define VIDEOREADER_H

#include <libv4l2.h>
#include <linux/videodev2.h>
#include <vector>


// Handy structure to return saying where the memory mapped region is
struct VideoReaderBuffer {
    __u32 idx;
    void* start;
    size_t length;
};

class VideoReader {
    // Class for reading from the webcam.  The idea is that it should be 
    // able to write into a user buffer.  If that user buffer happens to
    // be memory mapped into a PBO of the graphics card, so much the 
    // better...

private:

    int fd_ = -1;
    // File descriptor; by default, invalid

    int numBuffers_ = 0;
    // Number of buffers the streaming has allocated

    std::vector<int> dequeuedBufferIdxs_;
    std::vector<VideoReaderBuffer> activeMmaps_;
    // List of the buffers we have already dequeued.
    
    std::vector<VideoReaderBuffer> mmapBuffers(int numBuffers);
    void unmmapBuffers(std::vector<VideoReaderBuffer>& buffers);

public:

    // Move resources from another
    VideoReader(VideoReader&&);

    VideoReader(const char* filename, int width, int height);
    ~VideoReader();

    void startCapture();
    // Start streaming in

    void stopCapture();
    // Start streaming in
    
    VideoReaderBuffer getFrame(); 
    // Return the memory-mapped region for a frame

    void returnBuffer(const VideoReaderBuffer&);
    // Return a buffer that has previously been dequeued, so that the driver
    // can start using them again
    
    void returnBuffers();
    // Return buffers that have previously been dequeued, so that the driver
    // can start using them again
};

#endif

