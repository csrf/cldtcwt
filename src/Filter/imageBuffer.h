#ifndef IMAGE_BUFFER_H
#define IMAGE_BUFFER_H

#include <algorithm>

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
    ImageBuffer(cl::Context context,
                cl_mem_flags flags,
                size_t width, size_t height,
                size_t padding, size_t alignment);

    cl::Buffer buffer() const;

    size_t start() const;
    // Linear index to the upper left corner of the image

    size_t width() const;
    size_t height() const;
    size_t padding() const;
    size_t stride() const;


    void write(cl::CommandQueue& cq,
        const MemType* input,
        const std::vector<cl::Event> events = {},
        cl::Event* done = nullptr) const;


    void read(cl::CommandQueue& cq,
        MemType* output,
        const std::vector<cl::Event> events = {}) const;

private:
    cl::Buffer buffer_;


    size_t start_;
    // Location of the upper-left pixel in the image, linearly through
    // the buffer

    size_t width_;
    size_t padding_;
    size_t stride_;
    size_t height_;

};




template <typename MemType>
ImageBuffer<MemType>::ImageBuffer(cl::Context context,
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

    // Record the location of the upper left pixel, linear index
    // into the buffer
    start_ = stride_ * padding_ + padding_; 

    buffer_ = cl::Buffer(context, flags,
             stride_ * fullHeight * ImageElementTraits<MemType>::size);
}

#include <iostream>

template <typename MemType>
void ImageBuffer<MemType>::write(cl::CommandQueue& cq,
                const MemType* input,
                const std::vector<cl::Event> events,
                cl::Event* done) const
{
    const size_t numElements = buffer_.getInfo<CL_MEM_SIZE>() 
                            / ImageElementTraits<MemType>::size;

    std::vector<MemType> bufferContents(numElements);

    // Copy into the output buffer row by row
    auto writePos = bufferContents.begin() + start_;
    for (int n = 0; n < height_; ++n, writePos += stride_,
                                 input += width_) 
        std::copy(input, input + width_, writePos);

    // Read the internal contents of the buffer
    cq.enqueueWriteBuffer(buffer_, CL_TRUE, 
                         0, buffer_.getInfo<CL_MEM_SIZE>(),
                         &bufferContents[0],
                         &events, nullptr);
#if 0
    cq.enqueueWriteBufferRect(input.buffer(), CL_TRUE,
              makeCLSizeT<3>({sizeof(float) * input.padding(),
                              input.padding(), 0}),
              makeCLSizeT<3>({0,0,0}),
              makeCLSizeT<3>({input.width() * sizeof(float),
                              input.height(), 1}),
              input.stride() * sizeof(float), 0,
              0, 0,
              &inValues[0]);
#endif

}


template <typename MemType>
void ImageBuffer<MemType>::read(cl::CommandQueue& cq,
        MemType* output,
        const std::vector<cl::Event> events) const
{
    const size_t numElements = buffer_.getInfo<CL_MEM_SIZE>() 
                            / ImageElementTraits<MemType>::size;

    std::vector<MemType> bufferContents(numElements);
    
    // Read the internal contents of the buffer
    cq.enqueueReadBuffer(buffer_, CL_TRUE, 
                         0, buffer_.getInfo<CL_MEM_SIZE>(),
                         &bufferContents[0],
                         &events, nullptr);

    // Copy into the results row by row
    auto readPos = bufferContents.begin() + padding_ * stride_
                    + padding_;
    for (int n = 0; n < height_; ++n, readPos += stride_,
                                 output += width_) 
        std::copy(readPos, readPos + width_, output);

#if 0
    // This should work, but doesn't at the moment due to an AMD
    // bug
    cq.enqueueReadBufferRect(output.buffer(), CL_TRUE,
              makeCLSizeT<3>({sizeof(float) * output.padding(),
                             output.padding(), 0}),
              makeCLSizeT<3>({0,0,0}),
              makeCLSizeT<3>({output.width() * sizeof(float),
                             output.height(), 1}),
              output.stride() * sizeof(float), 0,
              0, 0,
              &outValues[0]);
#endif


}



template <typename MemType>
cl::Buffer ImageBuffer<MemType>::buffer() const
{
    return buffer_;
}




template <typename MemType>
size_t ImageBuffer<MemType>::start() const
{
    return start_;
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


// Complex pair is a type which we could use a good deal
template <typename Type>
struct __attribute__((packed)) Complex {
    Type real;
    Type imag;
};




#endif

