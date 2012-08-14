#include "clUtil.h"
#include <iostream>
#include <fstream>


int roundWGs(int l, int lWG)
{
    return lWG * (l / lWG + ((l % lWG) ? 1 : 0)); 
}



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


void saveRealImage(std::string filename,
                   cl::CommandQueue& cq, cl::Image2D& image)
{
    const size_t width = image.getImageInfo<CL_IMAGE_WIDTH>(),
                height = image.getImageInfo<CL_IMAGE_HEIGHT>();
    float output[height][width];
    readImage2D(cq, &output[0][0], image);

    // Open the file for output
    std::ofstream out(filename, std::ios_base::trunc | std::ios_base::out);

    // Produce the output in a file readable by MATLAB dlmread
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            out << output[y][x] << ((x+1) < width? "," : "");
        }

        if ((y+1) < height)
            out << "\n";
    }
}


void saveComplexImage(std::string filename,
                      cl::CommandQueue& cq, cl::Image2D& image)
{
    const size_t width = image.getImageInfo<CL_IMAGE_WIDTH>(),
                height = image.getImageInfo<CL_IMAGE_HEIGHT>();
    float output[height][width][2];
    readImage2D(cq, &output[0][0][0], image);

    // Open the file for output
    std::ofstream out(filename, std::ios_base::trunc | std::ios_base::out);

    // Produce the output in a file readable by MATLAB dlmread
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            out << output[y][x][0];
            if (output[y][x][1] >= 0)
                out << "+";
            out << output[y][x][1] << "j"
                << ((x+1) < width? "," : "");
        }

        if ((y+1) < height)
            out << "\n";
    }
}


void saveComplexBuffer(std::string filename,
                      cl::CommandQueue& cq, cl::Buffer& buffer)
{
    std::vector<float> output = readBuffer<float>(cq, buffer);

    // Open the file for output
    std::ofstream out(filename, std::ios_base::trunc | std::ios_base::out);

    // Produce the output in a file readable bn MATLAB dlmread
    for (size_t n = 0; n < output.size(); n += 2) {
        out << output[n];
        if (output[n+1] >= 0)
            out << "+";
        out << output[n+1] << "j";

        if ((n+2) < output.size())
            out << "\n";
    }
}

void displayRealImage(cl::CommandQueue& cq, cl::Image2D& image)
{
    const size_t width = image.getImageInfo<CL_IMAGE_WIDTH>(),
                height = image.getImageInfo<CL_IMAGE_HEIGHT>();
    float output[height][width];
    readImage2D(cq, &output[0][0], image);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x)
            std::cout << output[y][x] << "\t"; 

        std::cout << std::endl;
    }

    std::cout << std::endl;
}


