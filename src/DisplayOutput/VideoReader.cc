#include "VideoReader.h"

#include <stdexcept>
#include <fcntl.h>
#include <sys/mman.h>
#include <cstring>
#include <iostream>
#include <algorithm>



VideoReader::VideoReader(VideoReader&& m)
{
    // Make sure we have nothing open
    if (fd_ != -1)
        v4l2_close(fd_);

    // Transfer
    fd_ = m.fd_;
    numBuffers_ = m.numBuffers_;

    // Remove ownership from the other
    m.fd_ = -1;
}



VideoReader::VideoReader(const char* filename, int width, int height)
{
    // Open the video device
    fd_ = v4l2_open(filename, O_RDWR | O_NONBLOCK);

    // Check it worked
    if (fd_ == -1) 
        throw std::runtime_error("Failed to open V4L2 device");

    // Create a clear format, to say what output we want
    v4l2_format format;
    memset(&format, 0, sizeof(format));
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = width;
    format.fmt.pix.height = height;
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_YVU420;
    format.fmt.pix.field = V4L2_FIELD_NONE;

    if (v4l2_ioctl(fd_, VIDIOC_S_FMT, &format) == -1)
        throw std::runtime_error("Failed to set V4L2 device's format");

    if ((width != format.fmt.pix.width)
     || (height != format.fmt.pix.height)
     || (format.fmt.pix.pixelformat != V4L2_PIX_FMT_YVU420))
        throw std::logic_error("V4L2 did not match format requested");

    // Set up for streaming with user pointers
    v4l2_requestbuffers requestBuffers;
    memset(&requestBuffers, 0, sizeof(requestBuffers));
    requestBuffers.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    requestBuffers.memory = V4L2_MEMORY_MMAP;
    requestBuffers.count = 4;

    if (v4l2_ioctl(fd_, VIDIOC_REQBUFS, &requestBuffers) == -1) {
        if (errno == EINVAL)
            throw std::runtime_error("V4L2 failed requesting mmap streaming"
                                     " (not supported)");

        throw std::runtime_error("V4L2 failed requesting mmap streaming");
    }

    numBuffers_ = requestBuffers.count;
    std::cout << "num buffers " << numBuffers_ << "\n";

    activeMmaps_ = mmapBuffers(numBuffers_);
}


std::vector<VideoReaderBuffer> VideoReader::mmapBuffers(int numBuffers)
{
    std::vector<VideoReaderBuffer> mmaps;

    // Go through each, memory mapping and producing a list of the mapped
    // addresses
    for (int n = 0; n < numBuffers; ++n) {

        // Request info about this index buffer
        v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = n;

        if (v4l2_ioctl(fd_, VIDIOC_QUERYBUF, &buf) == -1)
            throw std::runtime_error("V4L2 failed to query buffer");


        // Use it to memory map
        VideoReaderBuffer returnBuffer = {
           buf.index,
           v4l2_mmap(nullptr, buf.length,
                     PROT_READ | PROT_WRITE, MAP_SHARED,
                     fd_, buf.m.offset),
           buf.length
        };

        if (returnBuffer.start == MAP_FAILED)
            throw std::runtime_error("V4L2 memory mapping failed");

        mmaps.push_back(returnBuffer);
    }

    return mmaps;
}

void VideoReader::unmmapBuffers(std::vector<VideoReaderBuffer>& buffers)
{
    // Unmap the buffers

    // Unmap all their memory
    for (const auto& m: buffers) 
        v4l2_munmap(m.start, m.length);
}

VideoReader::~VideoReader()
{
    if (fd_ != -1) {
        unmmapBuffers(activeMmaps_);

        v4l2_close(fd_);
    }
}



void VideoReader::startCapture()
{
    // Set going, and enqueue some buffers to go
    for (int n = 0; n < numBuffers_; ++n) {

        // Set up the buffer properties
        v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.index = n;
        buf.memory = V4L2_MEMORY_MMAP;

        if (v4l2_ioctl(fd_, VIDIOC_QBUF, &buf) == -1)
            throw std::runtime_error("V4L2 failed to enqueue buffer");

    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (v4l2_ioctl(fd_, VIDIOC_STREAMON, &type) == -1)
        throw std::runtime_error("V4L2 failed to start streaming");
}



void VideoReader::stopCapture()
{
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (v4l2_ioctl(fd_, VIDIOC_STREAMOFF, &type) == -1)
        throw std::runtime_error("V4L2 failed to stop streaming");
}



VideoReaderBuffer VideoReader::getFrame()
{
    v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    int n = 0;
    // Wait for a buffer to become available
    while (v4l2_ioctl(fd_, VIDIOC_DQBUF, &buf) == -1) {
        ++n;
        if (errno != EAGAIN)
            throw std::runtime_error("V4L2 failed to dequeue buffer");


    }

    dequeuedBufferIdxs_.push_back(buf.index);

    return activeMmaps_[buf.index];
}


void VideoReader::returnBuffer(const VideoReaderBuffer& buffer)
{
    // Enqueue the index
    v4l2_buffer newBuf;
    memset(&newBuf, 0, sizeof(newBuf));
    newBuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    newBuf.index = buffer.idx;
    newBuf.memory = V4L2_MEMORY_MMAP;

    if (v4l2_ioctl(fd_, VIDIOC_QBUF, &newBuf) == -1)
        throw std::runtime_error("V4L2 failed to enqueue buffer");
    
    auto pos = std::find(dequeuedBufferIdxs_.begin(),
                         dequeuedBufferIdxs_.end(),
                         buffer.idx);

    dequeuedBufferIdxs_.erase(pos);

}


void VideoReader::returnBuffers()
{
    // Put them all back on the queue
    for (int idx: dequeuedBufferIdxs_) {

        // Enqueue the index
        v4l2_buffer newBuf;
        memset(&newBuf, 0, sizeof(newBuf));
        newBuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        newBuf.index = idx;
        newBuf.memory = V4L2_MEMORY_MMAP;

        if (v4l2_ioctl(fd_, VIDIOC_QBUF, &newBuf) == -1)
            throw std::runtime_error("V4L2 failed to enqueue buffer");
    }

    dequeuedBufferIdxs_.clear();
}




