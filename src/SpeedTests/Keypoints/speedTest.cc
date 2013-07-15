// Copyright (C) 2013 Timothy Gale
#include "util/clUtil.h"
#include "CL/cl.hpp"
#include "DTCWT/intDtcwt.h"
#include "KeypointDetector/EnergyMaps/Eigen/energyMapEigen.h"
#include "KeypointDetector/EnergyMaps/EnergyMap/energyMap.h"
#include "KeypointDetector/peakDetector.h"
#include "KeypointDescriptor/extractDescriptors.h"

#include <chrono>


#include <ImfInputFile.h>
#include <ImfFrameBuffer.h>
#include <half.h>

#include <highgui.h>

std::tuple<cl::Platform, std::vector<cl::Device>, cl::Context> 
    initOpenCL();

typedef std::chrono::duration<double, std::milli>
    DurationMilliseconds;


struct Calculator {

    IntDtcwt dtcwt;

    EnergyMap energyMap;
    PeakDetector peakDetector;

    DescriptorExtracter descriptorExtracter;

    Calculator(cl::Context context,
               std::vector<cl::Device> devices);

};


Calculator::Calculator(cl::Context context,
                       std::vector<cl::Device> devices)
 :
    dtcwt {context, devices, 0.5},

    energyMap {context, devices},
    peakDetector {context, devices},

    descriptorExtracter {
        context, devices,
        peakDetector.getPosLength()
    }

{}



struct Workings {

    IntDtcwtOutput dtcwtOut;


    std::vector<float> scales; // List of the scale of each energy map, i.e. 
                               // how many pixels in the original image each
                               // pixel in the new image represents

    std::vector<cl::Image2D> energyMaps;
    std::vector<cl::Event> energyMapsDone;
    std::vector<cl::Image*> emPointers;

    size_t maxNumKeypoints;
    PeakDetectorResults peakDetectorResults;

    cl::Buffer descriptors;
    std::vector<cl::Event> descriptorsDone;

    Workings(cl::Context& context, 
             IntDtcwt& dtcwt,
             size_t width, size_t height, 
             size_t startLevel, size_t numLevels,
             PeakDetector& peakDetector, 
             size_t maxNumKeypoints, 
             size_t descriptorSize);

};


// Scale factors
static const std::vector<float> sf = {1.f, 7.f/8.f, 6.f/8.f, 5.f/8.f};

Workings::Workings(cl::Context& context, 
                   IntDtcwt& dtcwt,
                   size_t width, size_t height, 
                   size_t startLevel, size_t numLevels,
                   PeakDetector& peakDetector, 
                   size_t maxNumKeypointsVal, 
                   size_t descriptorSize)
    :
    dtcwtOut {
        dtcwt.createOutputs(width, height, startLevel, numLevels, sf)
    },

    maxNumKeypoints {maxNumKeypointsVal},

    peakDetectorResults {
       peakDetector.createResultsStructure(
                std::vector<size_t>(sf.size() * (numLevels - 1), 
                                    maxNumKeypoints),
                maxNumKeypoints
            )
    },

    descriptors {
        context, CL_MEM_READ_WRITE,
        maxNumKeypoints * descriptorSize
    },

    descriptorsDone { sf.size() * 2 * (numLevels-1) }
{
    
    // Create energy maps for each output level (other than the last,
    // which is only there for coarse detections)
    for (int i = 0; 
         i < (dtcwtOut.numTrees() * (dtcwtOut.numLevels() - 1));
         ++i) {
        energyMaps.push_back(
            createImage2D(context, 
                dtcwtOut[i].width(),
                dtcwtOut[i].height())
        );

        energyMapsDone.emplace_back();

        // Set up the scales (used in peak detection)
        scales.push_back(dtcwtOut.scale(i));
    }

    // Adapt to input format of peakDetector, which takes a list of pointers
    for (auto& e: energyMaps)
        emPointers.push_back(&e);
}



void detectKeypoints(cl::CommandQueue& commandQueue,
                     Calculator& calculator,
                     Workings& workings)
{

    // Calculate energy
    for (int l = 0; l < workings.energyMaps.size(); ++l) 
        calculator.energyMap(commandQueue, 
                  workings.dtcwtOut[l], 
                  workings.energyMaps[l], 
                  workings.dtcwtOut.doneEvents(l), 
                  &workings.energyMapsDone[l]);

    // Look for peaks
    calculator.peakDetector(commandQueue, workings.emPointers, workings.scales, 4.f, 0.f,
                               workings.peakDetectorResults,
                               workings.energyMapsDone);

}




void extractKeypoints(cl::CommandQueue& commandQueue,
                      Calculator& calculator,
                      Workings& workings)
{

    // Extract the descriptors
    for (size_t l = 0; l < workings.energyMaps.size(); ++l) {
        calculator.descriptorExtracter(commandQueue, 
                workings.dtcwtOut[l], workings.scales[l],      // Subband
                workings.dtcwtOut[l+sf.size()], workings.scales[l+sf.size()],  // Parent subband
                workings.peakDetectorResults.list(),         // Locations of keypoints
                workings.peakDetectorResults.cumCounts(), l, 
                workings.maxNumKeypoints, 
                        // Start indices within list of the different 
                        // levels; which level to extract; what the maximum
                        // number of keypoints we could be asking for is.
                workings.descriptors,
                workings.peakDetectorResults.listDone(),
                        // The cumulative counts rely on everything else
                        // in the peak detector being done
                &workings.descriptorsDone[l], &workings.descriptorsDone[l + workings.energyMaps.size()]
                        // Wait for both coarse and fine to be done
                );
    }


}



size_t getNumKeypoints(cl::CommandQueue& cq,
                       PeakDetectorResults& results)
{

    std::vector<cl::Event> waitEvents = {results.cumCountsDone()};

    cl_uint result;

    cq.enqueueReadBuffer(
        results.cumCounts(),
        CL_TRUE, // Block until read complete
        results.numLevels() * sizeof(cl_uint), sizeof(cl_uint),
            // Location and length to read
        &result,
        &waitEvents);

    return result;
}


#include <sstream>

template <typename T>
T readStr(const char* string)
{
    std::istringstream s(string);

    T result;
    s >> result;
    return result;
}




int main(int argc, char** argv)
{
    int numLevels = 4;
    int startLevel = 2;
    size_t maxNumKeypoints = 1000;
    size_t numIterations = 1000;

    std::string imageFilename;

    // First argument: image name
    if (argc >= 2) 
        imageFilename = readStr<std::string>(argv[1]);

    // Second argument: start level
    if (argc >= 3) 
        startLevel = readStr<size_t>(argv[2]);

    // Third argument: number of levels to calculate
    if (argc >= 4) 
        numLevels = readStr<size_t>(argv[3]);

    // Fourth argument: number of iterations
    if (argc >= 5) 
        numIterations = readStr<size_t>(argv[4]);

    // Fifth argument: maximum number of keypoints
    if (argc >= 6) 
        maxNumKeypoints = readStr<size_t>(argv[5]);


    // Read the image in
    cv::Mat bmp = cv::imread(argv[1], 0);
    cv::Mat floatBmp;
    bmp.convertTo(floatBmp, CV_32F);

    size_t width = floatBmp.cols, height = floatBmp.rows;

    cl::Platform platform;
    std::vector<cl::Device> devices;
    cl::Context context;

    std::tie(platform, devices, context) = initOpenCL();
    cl::Device device {devices[0]};

    cl::CommandQueue commandQueue {context, device}; 

    Calculator calculator {context, {device}};
                              
    Workings workings {
        context, 
        calculator.dtcwt,
        width, height, 
        startLevel, numLevels,
        calculator.peakDetector,
        maxNumKeypoints,
        calculator.descriptorExtracter.getNumFloatsInDescriptor()
             * sizeof(cl_float)
    };


    cl::Image2D input {
        context,
        CL_MEM_READ_WRITE,
        cl::ImageFormat {CL_LUMINANCE, CL_FLOAT},
        width, height, 0
    };

    cl::Event inputReady;
    commandQueue.enqueueWriteImage(input, CL_TRUE, 
                                   makeCLSizeT<3>({0, 0, 0}), 
                                   makeCLSizeT<3>({width, height, 1}), 
                                   0, 0,
                                   floatBmp.ptr());
    commandQueue.finish();

    auto t1 = std::chrono::steady_clock::now();
    
    // Transform
    for (int n = 0; n < numIterations; ++n)
        calculator.dtcwt(commandQueue, input, workings.dtcwtOut);


    commandQueue.finish();
    auto t2 = std::chrono::steady_clock::now();

    for (int n = 0; n < numIterations; ++n)
        detectKeypoints(commandQueue, calculator, workings);

    commandQueue.finish();
    auto t3 = std::chrono::steady_clock::now();



    for (int n = 0; n < numIterations; ++n)
        extractKeypoints(commandQueue, calculator, workings);
 
    commandQueue.finish();
    auto t4 = std::chrono::steady_clock::now();

    std::cout << "Dimensions: " << width << "x" << height << std::endl;

    std::cout << "Keypoints: "
              << getNumKeypoints(commandQueue, workings.peakDetectorResults)
              << std::endl;

    std::cout 
      << "DTCWT: "
      << DurationMilliseconds(t2 - t1).count() / numIterations << "ms\n"
      << "Keypoint detection: "
      << DurationMilliseconds(t3 - t2).count() / numIterations << "ms\n"
      << "Keypoint extraction: "
      << DurationMilliseconds(t4 - t3).count() / numIterations << "ms\n";

    return 0;
}




std::tuple<cl::Platform, std::vector<cl::Device>, cl::Context> 
    initOpenCL()
{
    // Get platform, devices

    // Retrive platform information
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.size() == 0)
        throw std::runtime_error("No platforms!");

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);

    // Create a context to work in 
    cl::Context context(devices);

    return std::make_tuple(platforms[0], devices, context);
}


