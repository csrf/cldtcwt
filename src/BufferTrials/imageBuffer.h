#ifndef IMAGE_BUFFER_H
#define IMAGE_BUFFER_H


#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"


class ImageBuffer {
    // To make it easier to create an image buffer with sufficient
    // padding on all sides, that can have the desired alignment
    // between rows.

public:
    ImageBuffer() = default;
    ImageBuffer(const ImageBuffer&) = default;
    ImageBuffer(cl::Context& context,
                cl_mem_flags flags,
                size_t width, size_t height,
                size_t padding, size_t alignment);

    cl::Buffer buffer() const;
    size_t width() const;
    size_t height() const;
    size_t padding() const;
    size_t stride() const;

private:
    cl::Buffer buffer_;
    size_t width_;
    size_t padding_;
    size_t stride_;
    size_t height_;

};





#endif

