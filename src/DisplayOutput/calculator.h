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

    cl::Buffer numKps;
    cl::Buffer keypointLocs;

public:

    Calculator(const Calculator&) = default;
    Calculator() = default;

    Calculator(cl::Context& context,
               const std::vector<cl::Device>& devices,
               int width, int height);

    void operator() (const cl::Image& input);

    std::vector<::LevelOutput*> levelOutputs(void);

    std::vector<cl::Buffer*> keypointLocations(void);
    cl::Buffer* keypointCounts(void);
    std::vector<cl::Event> keypointLocationEvents(void);

};



#endif 

