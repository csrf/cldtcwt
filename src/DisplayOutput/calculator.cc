#include "calculator.h"
#include "util/clUtil.h"


Calculator::Calculator(cl::Context& context,
                       const cl::Device& device,
                       int width, int height,
                       int maxNumKeypointsPerLevel)
 :  commandQueue(context, device),
    dtcwt(context, {device}, commandQueue),
    abs(context, {device}),
    energyMap(context, {device}),
    findMax(context, {device})
{
    const int numLevels = 6;
    const int startLevel = 1;

    // Create the DTCWT, temporaries and outputs
    dtcwtTemps = dtcwt.createContext(width, height,
                                     numLevels, startLevel);
    dtcwtOut = DtcwtOutput(dtcwtTemps);

    // Create energy maps for each output level (other than the last,
    // which is only there for coarse detections)
    for (int l = 0; l < (dtcwtOut.subbands.size() - 1); ++l) {
        energyMaps.push_back(
            createImage2D(context, 
                dtcwtOut.subbands[l].sb[0].getImageInfo<CL_IMAGE_WIDTH>(),
                dtcwtOut.subbands[l].sb[0].getImageInfo<CL_IMAGE_HEIGHT>())
        );

        energyMapsDone.push_back(cl::Event());
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

    // Create store space for how many keypoints were detected at each level 
    std::vector<float> zeroV = {0.f};
    keypointCounts_ = createBuffer(context, commandQueue, zeroV);

    // And for keypoint locations
    std::vector<float> kpLocsV(2*maxNumKeypointsPerLevel);
    keypointLocs_ = createBuffer(context, commandQueue, kpLocsV);
}



void Calculator::operator() (cl::Image& input,
                             const std::vector<cl::Event>& waitEvents)
{
    // Transform
    dtcwt(commandQueue, input, dtcwtTemps, dtcwtOut);

    // Calculate energy
    for (int l = 0; l < energyMaps.size(); ++l)
        energyMap(commandQueue, dtcwtOut.subbands[l], 
                                energyMaps[l], &energyMapsDone[l]);

    // Clear the peak counts
    std::vector<char> zeros(keypointCounts_.getInfo<CL_MEM_SIZE>(), 0);
    cl::Event clearDone;
    writeBuffer(commandQueue, keypointCounts_, zeros, &clearDone);

    // Look for peaks
    findMax(commandQueue, energyMaps[0], zeroImage, zeroImage, 0.1f,
                          keypointLocs_, keypointCounts_,
                          {energyMapsDone[0], clearDone},
                          &findMaxDone_);
}


cl::Image2D Calculator::getEnergyMapLevel2()
{
    return energyMaps[0];
}


std::vector<LevelOutput*> Calculator::levelOutputs(void)
{
    std::vector<LevelOutput*> outputs;

    for (auto& l: dtcwtOut.subbands)
        outputs.push_back(&l);
    
    return outputs;
}



std::vector<cl::Buffer*> Calculator::keypointLocations(void)
{
    std::vector<cl::Buffer*> locations;
    locations.push_back(&keypointLocs_);
    return locations;
}


cl::Buffer* Calculator::keypointCounts(void)
{
    return &keypointCounts_;
}


std::vector<cl::Event> Calculator::keypointLocationEvents(void)
{
    std::vector<cl::Event> doneEvents;

    doneEvents.push_back(findMaxDone_);

    return doneEvents;
}




