#include "calculator.h"


Calculator::Calculator(cl::Context& context,
                       const cl::Device& device,
                       int width, int height)
{
    // Work with our own command queue
    commandQueue = cl::CommandQueue(context, device);

    const int numLevels = 6;
    const int startLevel = 1;

    // Create the DTCWT, temporaries and outputs
    dtcwt = Dtcwt(context, {device}, commandQueue);
    dtcwtTemps = dtcwt.createContext(width, height,
                                     numLevels, startLevel);
    out = DtcwtOutput(dtcwtTemps);

    // Create energy maps for each output level (other than the last,
    // which is only there for coarse detections)
    for (int l = 0; l < (dtcwtOut.subbands.size() - 1); ++l) {
        energyMaps.push_back(
            createImage2D(context, 
                dtcwtOut.subbands[l].sb[0].getImageInfo<CL_IMAGE_WIDTH>(),
                dtcwtOut.subbands[l].sb[0].getImageInfo<CL_IMAGE_HEIGHT>())
        );
    }

    // Zero image for use off the ends of maximum finding
    float zerof = 0;
    zeroImage = {
        context, 
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        cl::ImageFormat(CL_LUMINANCE, CL_FLOAT), 
        1, 1, 0,
        &zerof
    };

    // Create the kernels
    abs = Abs(context, {device});
    energyMap = EnergyMap(context, {device});
    findMax = FindMax(context, {device});

    std::vector<float> zeroV = {0.f};
    numKps = createBuffer(context, commandQueue, zeroV);
 
}



