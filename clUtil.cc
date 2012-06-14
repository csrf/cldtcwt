#include "clUtil.h"

cl::Buffer createBuffer(cl::Context& context,
                        cl::CommandQueue& commandQueue, 
                        const std::vector<float>& data)
{
    //CL_MEM_COPY_HOST_PTR
    cl::Buffer buffer(context, CL_MEM_READ_WRITE,
                      sizeof(float) * data.size());

    commandQueue.enqueueWriteBuffer(buffer, CL_TRUE, 0,
                                    sizeof(float) * data.size(),
                                    &(*(data.begin())));
    commandQueue.finish();

    return buffer;
}



cl::Image2D createImage2D(cl::Context& context, 
                          int width, int height)
{
    cl::Image2D outImage(context,
                       0,
                       cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), 
                       width, height);

    return outImage;
}


void writeImage2D(cl::CommandQueue& commandQueue,
                  cl::Image2D& image, float* memory)
{
    // Write memory into image, blocking until done, using the
    // commmandQueue.

    // Get the dimensions from the image, rather than requiring the user to
    // pass it.
    const int width = image.getImageInfo<CL_IMAGE_WIDTH>();
    const int height = image.getImageInfo<CL_IMAGE_HEIGHT>();

    // Set up where and how far to read
    cl::size_t<3> origin, extents;
    origin.push_back(0);
    origin.push_back(0);
    origin.push_back(0);
    extents.push_back(width);
    extents.push_back(height);
    extents.push_back(1);

    // Perform the write, blocking
    commandQueue.enqueueWriteImage(image, CL_TRUE, origin, extents, 0, 0,
                                   memory);
}


void readImage2D(cl::CommandQueue& commandQueue,
                 float* outMemory, cl::Image2D& image)
{
    // Read the image into outMemory, blocking until done, using the
    // commmandQueue.

    // Get the dimensions from the image, rather than requiring the user to
    // pass it.
    const int width = image.getImageInfo<CL_IMAGE_WIDTH>();
    const int height = image.getImageInfo<CL_IMAGE_HEIGHT>();

    // Set up where and how far to read
    cl::size_t<3> origin, extents;
    origin.push_back(0);
    origin.push_back(0);
    origin.push_back(0);
    extents.push_back(width);
    extents.push_back(height);
    extents.push_back(1);

    // Do the read, blocking style
    commandQueue.enqueueReadImage(image, CL_TRUE, origin, extents,
                                  0, 0, outMemory);
}


cl::Image2D createImage2D(cl::Context& context, cv::Mat& mat)
{
    if (mat.type() == CV_32F) {
        // If in the right format already, just create the image and point
        // it to the data
        return cl::Image2D(context, 
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), 
                           mat.cols, mat.rows, 0,
                           mat.ptr());
    } else {
        // We need to get it into the right format first.  Convert then
        // send
        cv::Mat floatedMat;
        mat.convertTo(floatedMat, CV_32F);

        return cl::Image2D(context, 
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), 
                           floatedMat.cols, floatedMat.rows, 0,
                           floatedMat.ptr());
    }
}


