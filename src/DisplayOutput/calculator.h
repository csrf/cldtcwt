#ifndef CALCULATOR_H
#define CALCULATOR_H

#include <vector>
#include <CL/cl.hpp>

#include "DTCWT/dtcwt.h"
#include "Abs/abs.h"
#include "KeypointDetector/peakDetector.h"
#include "KeypointDetector/EnergyMaps/InterpMap/interpMap.h"
#include "KeypointDetector/EnergyMaps/InterpPhaseMap/interpPhaseMap.h"
#include "KeypointDescriptor/extractDescriptors.h"


class Calculator {

    // Takes an input image, and produces subbands and keypoint locations

private:

    cl::Platform platform;
    std::vector<cl::Device> devices;
    cl::Context context;
    cl::CommandQueue commandQueue; 

    Dtcwt dtcwt;
    Abs abs;
    InterpMapEigen energyMap;
    PeakDetector peakDetector;

    cl::Image2D zeroImage;

    DtcwtTemps dtcwtTemps;
    DtcwtOutput dtcwtOut;

    std::vector<cl::Image2D> energyMaps;
    std::vector<cl::Event> energyMapsDone;

    size_t maxNumKeypoints_;

    PeakDetectorResults peakDetectorResults;
    std::vector<float> scales; // List of the scale of each energy map, i.e. 
                               // how many pixels in the original image each
                               // pixel in the new image represents

    // Output of descriptors
    cl::Buffer descriptors_;
    std::vector<cl::Event> descriptorsDone_;

    DescriptorExtracter descriptorExtracter_;

public:

    Calculator(const Calculator&) = default;
    Calculator() = default;

    Calculator(cl::Context& context,
               const cl::Device& device,
               int width, int height,
               int maxNumKeypoints = 1000);

    void operator() (ImageBuffer<cl_float>& input, 
                     const std::vector<cl::Event>& waitEvents = {});

    std::vector<::LevelOutput*> levelOutputs(void);

    cl::Image2D getEnergyMapLevel2(void);
    cl::Buffer keypointLocations(void);
    size_t numFloatsPerKPLocation(void);
    cl::Buffer keypointCumCounts(void);
    std::vector<cl::Event> keypointLocationEvents(void);

};



#endif 

