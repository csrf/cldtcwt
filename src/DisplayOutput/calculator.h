#ifndef CALCULATOR_H
#define CALCULATOR_H

#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include "DTCWT/dtcwt.h"
#include "Abs/abs.h"
#include "KeypointDetector/peakDetector.h"
#include "KeypointDetector/EnergyMaps/Eigen/energyMapEigen.h"
#include "KeypointDetector/EnergyMaps/EnergyMap/energyMap.h"
#include "KeypointDetector/EnergyMaps/BTK/energyMapBTK.h"
#include "KeypointDetector/EnergyMaps/CrossProduct/crossProduct.h"
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
    CrossProductMap energyMap;
    //EnergyMap energyMap;
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

    std::vector<::Subbands*> levelOutputs(void);
    std::vector<std::vector<cl::Event>> levelDoneEvents(void) const;

    cl::Image2D getEnergyMapLevel2(void);
    cl::Buffer keypointLocations(void);
    cl::Buffer keypointDescriptors(void);
    size_t numFloatsPerKPLocation(void);
    cl::Buffer keypointCumCounts(void);
    std::vector<cl::Event> keypointLocationEvents(void);

};



#endif 

