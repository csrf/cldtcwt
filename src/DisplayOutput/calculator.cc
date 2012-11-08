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
    peakDetector(context, {device})
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


    // Create the temporaries and results for peak detection
    peakDetectorResults = peakDetector.createResultsStructure
        (std::vector<size_t>(energyMaps.size(), 1000), 2000);
    // i.e. allow 1000 keypoints at any given level, and up to 2000 overall.

    // Set up the scales (used in peak detection)
    float s = 4.0f;
    for (int n = 0; n < energyMaps.size(); ++n) {
        scales.push_back(s);
        s *= 2.0f;
    }


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

    // Adapt to input format of peakDetector, which takes a list of pointers
    std::vector<cl::Image*> emPointers;
    for (auto& e: energyMaps)
        emPointers.push_back(&e);

    // Look for peaks
    peakDetector(commandQueue, emPointers, scales, 0.1f,
                               peakDetectorResults,
                               energyMapsDone);
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



cl::Buffer Calculator::keypointLocations(void)
{
    return peakDetectorResults.list;
}


cl::Buffer Calculator::keypointCumCounts(void)
{
    return peakDetectorResults.cumCounts;
}


std::vector<cl::Event> Calculator::keypointLocationEvents(void)
{
    return peakDetectorResults.listDone;
}




