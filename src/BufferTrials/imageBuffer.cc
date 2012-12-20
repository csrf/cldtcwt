#include "imageBuffer.h"



ImageBuffer::ImageBuffer(cl::Context& context,
                         cl_mem_flags flags,
                         size_t width, size_t height,
                         size_t padding, size_t alignment)
    : width_(width),
      height_(height),
      padding_(padding),
      stride_(width + 2*padding)
{
    // Stride might need extending to respect alignment
    size_t overshoot = stride_ % alignment;
    if (overshoot != 0)
        stride_ += alignment - overshoot;

    // Pad the height to meet the alignment as well, in case
    // anyone ever tried to access that region of memory
    size_t fullHeight = height_ + 2*padding_;
    overshoot = fullHeight % alignment;
    if (overshoot)
        fullHeight += alignment - overshoot;

    buffer_ = cl::Buffer(context, flags,
                         stride_ * fullHeight * sizeof(float));
}




cl::Buffer ImageBuffer::buffer() const
{
    return buffer_;
}


size_t ImageBuffer::width() const
{
    return width_;
}


size_t ImageBuffer::height() const
{
    return height_;
}



size_t ImageBuffer::padding() const
{
    return padding_;
}



size_t ImageBuffer::stride() const
{
    return stride_;
}







