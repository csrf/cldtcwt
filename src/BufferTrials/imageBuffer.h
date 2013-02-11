#ifndef IMAGE_BUFFER_H
#define IMAGE_BUFFER_H


#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"


// ImageElementTraits
// 
// For each class of value, tells ImageBuffer how to store it: how much
// memory an element takes up, and what kind of pointer to assume.
//
// By default, this is a pointer to the type and its sizeof(.).  However,
// in some cases this won't work, e.g. for a half or something which doesn't
// have a particular type.  In that case, void can be used as the type, but the
// length can be correct; specialisation is necessary

template <typename MemType>
struct ImageElementTraits {
    typedef MemType Type;
    static const size_t size = sizeof(Type);
};


template <typename MemType>
class ImageBuffer {
    // To make it easier to create an image buffer with sufficient
    // padding on all sides, that can have the desired alignment
    // between rows.

public:
    ImageBuffer() = default;
    ImageBuffer(const ImageBuffer<MemType>&) = default;
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




template <typename MemType>
ImageBuffer<MemType>::ImageBuffer(cl::Context& context,
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
             stride_ * fullHeight * ImageElementTraits<MemType>::size);
}



template <typename MemType>
cl::Buffer ImageBuffer<MemType>::buffer() const
{
    return buffer_;
}


template <typename MemType>
size_t ImageBuffer<MemType>::width() const
{
    return width_;
}


template <typename MemType>
size_t ImageBuffer<MemType>::height() const
{
    return height_;
}



template <typename MemType>
size_t ImageBuffer<MemType>::padding() const
{
    return padding_;
}



template <typename MemType>
size_t ImageBuffer<MemType>::stride() const
{
    return stride_;
}



#endif

