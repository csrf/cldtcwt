#ifndef CLUTIL_H
#define CLUTIL_H


#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "cl.hpp"

#include <vector>

#include <highgui.h>
#include <stdexcept>

int roundWGs(int l, int lWG);

cl::Buffer createBuffer(cl::Context&, cl::CommandQueue&,
                        const std::vector<float>& data);

template <typename T>
void writeBuffer(cl::CommandQueue& cq, const cl::Buffer& buffer,
                 const std::vector<T>& vals)
{
    cq.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(T) * vals.size(),
                          &vals[0]);
}


template <typename T>
std::vector<T> readBuffer(cl::CommandQueue& cq, const cl::Buffer& buffer)
{
    const size_t size = buffer.getInfo<CL_MEM_SIZE>();
    std::vector<T> vals(size / sizeof(T));

    cq.enqueueReadBuffer(buffer, CL_TRUE, 0, size, &vals[0]);

    return vals;
}



cl::Image2D createImage2D(cl::Context&, int width, int height);

cl::Image2D createImage2D(cl::Context& context, cv::Mat& mat);

void writeImage2D(cl::CommandQueue& commandQueue,
                  cl::Image2D& image, float* memory);

void readImage2D(cl::CommandQueue& commandQueue,
                 float* outMemory, cl::Image2D& image);


void saveRealImage(std::string filename,
                   cl::CommandQueue& cq, cl::Image2D& image);

void saveComplexImage(std::string filename,
                      cl::CommandQueue& cq, cl::Image2D& image);

void saveComplexBuffer(std::string filename,
                       cl::CommandQueue& cq, cl::Buffer& buffer);

void displayRealImage(cl::CommandQueue& cq, cl::Image2D& image);


class CLContext {
public:

    CLContext()
    {
        // Get platform, devices, then create a context

        // Retrive platform information
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.size() == 0)
            throw std::runtime_error("No platforms!");

        // Use the first platform
        platform = platforms[0];

        platform.getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);

        // Create a context to work in 
        context = cl::Context(devices);
    }

    cl::Platform platform;
    std::vector<cl::Device> devices;
    cl::Context context;

};


#endif
