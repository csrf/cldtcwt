#ifndef CALCULATOR_H
#define CALCULATOR_H

#include <vector>
#include <CL/cl.hpp>

#include "DTCWT/dtcwt.h"
#include "MiscKernels/abs.h"
#include "KeypointDetector/findMax.h"


class Calculator {

    // Takes an input image, and produces subbands and keypoint locations

private:

    cl::Platform platform;
    std::vector<cl::Device> devices;
    cl::Context context;
    cl::CommandQueue commandQueue; 

    Dtcwt dtcwt;
    Abs abs;
    EnergyMap energyMap;
    FindMax findMax;

    cl::Image2D zeroImage;

    DtcwtTemps dtcwtTemps;
    DtcwtOutput dtcwtOut;

    std::vector<cl::Image2D> energyMaps;
    std::vector<cl::Event> energyMapsDone;

    cl::Buffer keypointCounts_; // per level, how many were found
    cl::Buffer keypointLocs_;   // where were they?

    cl::Event findMaxDone_;  // Whether the keypoint detection has finished

public:

    Calculator(const Calculator&) = default;
    Calculator() = default;

    Calculator(cl::Context& context,
               const cl::Device& device,
               int width, int height,
               int maxNumKeypointsPerLevel = 1000);

    void operator() (cl::Image& input, 
                     const std::vector<cl::Event>& waitEvents = {});

    std::vector<::LevelOutput*> levelOutputs(void);

    cl::Image2D getEnergyMapLevel2(void);
    std::vector<cl::Buffer*> keypointLocations(void);
    cl::Buffer* keypointCounts(void);
    std::vector<cl::Event> keypointLocationEvents(void);

};



#endif 

