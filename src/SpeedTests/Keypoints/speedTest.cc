#include "util/clUtil.h"
#include "CL/cl.hpp"
#include "DTCWT/dtcwt.h"
#include "KeypointDetector/EnergyMaps/Eigen/energyMapEigen.h"
#include "KeypointDetector/EnergyMaps/InterpMap/interpMap.h"
#include "KeypointDetector/peakDetector.h"
#include "KeypointDescriptor/extractDescriptors.h"

#include <chrono>

std::tuple<cl::Platform, std::vector<cl::Device>, cl::Context> 
    initOpenCL();

typedef std::chrono::duration<double, std::milli>
    DurationMilliseconds;



int main()
{
    const int width = 1280, height = 720;
    const int numLevels = 3;
    const int startLevel = 2;
    const size_t maxNumKeypoints = 1000;



    cl::Platform platform;
    std::vector<cl::Device> devices;
    cl::Context context;

    std::tie(platform, devices, context) = initOpenCL();
    cl::Device device {devices[0]};

    cl::CommandQueue commandQueue {context, device}; 

    Dtcwt dtcwt {context, {device}, 0.5};
    InterpMapEigen energyMap {context, {device}};
    PeakDetector peakDetector {context, {device}};

    DescriptorExtracter descriptorExtracter {
        context, {device},
        peakDetector.getPosLength()
    };


    DtcwtTemps dtcwtTemps {context, width, height, startLevel, numLevels};;
    DtcwtOutput dtcwtOut {dtcwtTemps.createOutputs()};

    std::vector<cl::Image2D> energyMaps;
    std::vector<cl::Event> energyMapsDone;

    std::vector<float> scales; // List of the scale of each energy map, i.e. 
                               // how many pixels in the original image each
                               // pixel in the new image represents
                               
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

    PeakDetectorResults peakDetectorResults 
      = peakDetector.createResultsStructure(
                std::vector<size_t>(energyMaps.size(), maxNumKeypoints),
                maxNumKeypoints
            );
 
    // Output of descriptors
    cl::Buffer descriptors {
        context, CL_MEM_READ_WRITE,
        maxNumKeypoints * descriptorExtracter.getNumFloatsInDescriptor()
            * sizeof(cl_uint),
    };

    std::vector<cl::Event> descriptorsDone_;



 

    // Transform
    //dtcwt(commandQueue, input, dtcwtTemps, dtcwtOut, {}/* wait events */);

    // Calculate energy
    for (int l = 0; l < energyMaps.size(); ++l)
        energyMap(commandQueue, 
                  dtcwtOut.level(dtcwtOut.startLevel() + l), 
                  energyMaps[l], 
                  dtcwtOut.doneEvents(dtcwtOut.startLevel() + l), 
                  &energyMapsDone[l]);

    // Adapt to input format of peakDetector, which takes a list of pointers
    std::vector<cl::Image*> emPointers;
    for (auto& e: energyMaps)
        emPointers.push_back(&e);

    // Look for peaks
    peakDetector(commandQueue, emPointers, scales, 0.02, 0.f,
                               peakDetectorResults,
                               energyMapsDone);

    // Extract the descriptors
    for (size_t l = 0; l < (energyMaps.size() - 1); ++l) {
        descriptorExtracter(commandQueue, 
                dtcwtOut[l], scales[l],      // Subband
                dtcwtOut[l+1], scales[l+1],  // Parent subband
                peakDetectorResults.list,         // Locations of keypoints
                peakDetectorResults.cumCounts, l, maxNumKeypoints, 
                        // Start indices within list of the different 
                        // levels; which level to extract; what the maximum
                        // number of keypoints we could be asking for is.
                descriptors,
                {peakDetectorResults.cumCountsDone},
                        // The cumulative counts rely on everything else
                        // in the peak detector being done
                &descriptorsDone_[l], &descriptorsDone_[l + energyMaps.size()]
                        // Wait for both coarse and fine to be done
                );
    }
 

    

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


