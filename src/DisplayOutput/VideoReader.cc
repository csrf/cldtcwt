#include "VideoReader.h"

#include <stdexcept>
#include <fcntl.h>
#include <linux/videodev2.h>


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

#include <cstring>


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
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_BGR24;
    format.fmt.pix.field = V4L2_FIELD_NONE;

    if (v4l2_ioctl(fd_, VIDIOC_S_FMT, &format) == -1)
        throw std::runtime_error("Failed to set V4L2 device's format");

    if ((width != format.fmt.pix.width)
     || (height != format.fmt.pix.height)
     || (format.fmt.pix.pixelformat != V4L2_PIX_FMT_BGR24))
        throw std::logic_error("V4L2 did not match format requested");

    // Set up for streaming with user pointers
    v4l2_requestbuffers requestBuffers;
    memset(&requestBuffers, 0, sizeof(requestBuffers));
    requestBuffers.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    requestBuffers.memory = V4L2_MEMORY_MMAP;
    requestBuffers.count = 20;

    if (v4l2_ioctl(fd_, VIDIOC_REQBUFS, &requestBuffers) == -1) {
        if (errno == EINVAL)
            throw std::runtime_error("V4L2 failed requesting mmap streaming"
                                     " (not supported)");

        throw std::runtime_error("V4L2 failed requesting mmap streaming");
    }

    numBuffers_ = requestBuffers.count;

}



VideoReader::~VideoReader()
{
    if (fd_ != -1)
        v4l2_close(fd_);
}



void VideoReader::startCapture()
{
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (v4l2_ioctl(fd_, VIDIOC_STREAMON, &type) == -1)
        throw std::runtime_error("V4L2 failed to start streaming");
    // Set going, and quue
    for (int n = 0; n < numBuffers_; ++n) {
    }
}




