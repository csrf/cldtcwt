#include "util/clUtil.h"
#include "CL/cl.hpp"
#include "DTCWT/dtcwt.h"
#include "KeypointDetector/EnergyMaps/Eigen/energyMapEigen.h"
#include "KeypointDetector/EnergyMaps/InterpMap/interpMap.h"
#include "KeypointDetector/peakDetector.h"
#include "KeypointDescriptor/extractDescriptors.h"

#include <chrono>


#include <ImfInputFile.h>
#include <ImfFrameBuffer.h>
#include <half.h>

std::tuple<cl::Platform, std::vector<cl::Device>, cl::Context> 
    initOpenCL();

typedef std::chrono::duration<double, std::milli>
    DurationMilliseconds;


struct Calculator {

    Dtcwt dtcwt;

    InterpMapEigen energyMap;
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


std::vector<float> readY(const char filename[], 
                        size_t& width, size_t& height)
{
    Imf::InputFile file(filename);

    Imath::Box2i window = file.header().dataWindow();
    width = window.max.x - window.min.x + 1;
    height = window.max.y - window.min.y + 1;

    Imf::FrameBuffer frameBuffer;

    std::vector<float> values(width * height);

    frameBuffer.insert("Y", 
        Imf::Slice {
            Imf::FLOAT, 
            reinterpret_cast<char*>
                (&values[0] - window.min.x - window.min.y * width),
            sizeof(half), sizeof(half) * width,
            1, 1, 0.f
        });
                                  
    file.setFrameBuffer(frameBuffer);
    file.readPixels(window.min.y, window.max.y);

    return values;
}


struct Workings {

    DtcwtTemps dtcwtTemps;
    DtcwtOutput dtcwtOut;


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
             size_t width, size_t height, 
             size_t startLevel, size_t numLevels,
             PeakDetector& peakDetector, 
             size_t maxNumKeypoints, 
             size_t descriptorSize);

};




Workings::Workings(cl::Context& context, 
                   size_t width, size_t height, 
                   size_t startLevel, size_t numLevels,
                   PeakDetector& peakDetector, 
                   size_t maxNumKeypointsVal, 
                   size_t descriptorSize)
    :
    dtcwtTemps {context, width, height, startLevel, numLevels},
    dtcwtOut {dtcwtTemps.createOutputs()},

    maxNumKeypoints {maxNumKeypointsVal},

    peakDetectorResults {
       peakDetector.createResultsStructure(
                std::vector<size_t>(numLevels - 1, maxNumKeypoints),
                maxNumKeypoints
            )
    },

    descriptors {
        context, CL_MEM_READ_WRITE,
        maxNumKeypoints * descriptorSize
    },

    descriptorsDone { 2*(numLevels-1) }
{
    
    // Create energy maps for each output level (other than the last,
    // which is only there for coarse detections)
    float s = 2;
    for (int i = 0; i < (dtcwtOut.numLevels() - 1); ++i) {
        energyMaps.push_back(
            createImage2D(context, 
                dtcwtOut.level(dtcwtOut.startLevel() + i).width(),
                dtcwtOut.level(dtcwtOut.startLevel() + i).height())
        );

        energyMapsDone.emplace_back();

        // Set up the scales (used in peak detection)
        scales.push_back(s *= 2);
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
                  workings.dtcwtOut.level(workings.dtcwtOut.startLevel() + l), 
                  workings.energyMaps[l], 
                  workings.dtcwtOut.doneEvents(workings.dtcwtOut.startLevel() + l), 
                  &workings.energyMapsDone[l]);

    // Look for peaks
    calculator.peakDetector(commandQueue, workings.emPointers, workings.scales, 0.02, 0.f,
                               workings.peakDetectorResults,
                               workings.energyMapsDone);

}




void extractKeypoints(cl::CommandQueue& commandQueue,
                      Calculator& calculator,
                      Workings& workings)
{

    // Extract the descriptors
    for (size_t l = 0; l < (workings.energyMaps.size() - 1); ++l) {
        calculator.descriptorExtracter(commandQueue, 
                workings.dtcwtOut[l], workings.scales[l],      // Subband
                workings.dtcwtOut[l+1], workings.scales[l+1],  // Parent subband
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




int main()
{
    const int numLevels = 3;
    const int startLevel = 2;
    const size_t maxNumKeypoints = 1000;


    size_t width, height;
    std::vector<float> values = readY("logo.exr", width, height);


    cl::Platform platform;
    std::vector<cl::Device> devices;
    cl::Context context;

    std::tie(platform, devices, context) = initOpenCL();
    cl::Device device {devices[0]};

    cl::CommandQueue commandQueue {context, device}; 

    Calculator calculator {context, {device}};
                              
    Workings workings {
        context, 
        width, height, 
        startLevel, numLevels,
        calculator.peakDetector,
        maxNumKeypoints,
        calculator.descriptorExtracter.getNumFloatsInDescriptor()
             * sizeof(cl_float)
    };

    std::cout << width << "x" << height << std::endl;

    ImageBuffer<float> input {
        context, CL_MEM_READ_WRITE, 
        width, height,
        16, 32
    };

    cl::Event inputReady;
    input.write(commandQueue, &values[0], {}, &inputReady);
 
    commandQueue.finish();

    // Transform
    calculator.dtcwt(commandQueue, input, 
                     workings.dtcwtTemps, 
                     workings.dtcwtOut);


    commandQueue.finish();

    detectKeypoints(commandQueue, calculator, workings);

    commandQueue.finish();

    std::cout << getNumKeypoints(commandQueue, workings.peakDetectorResults)
              << std::endl;

    extractKeypoints(commandQueue, calculator, workings);
 
    commandQueue.finish();

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


