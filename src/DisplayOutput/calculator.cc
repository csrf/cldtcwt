#include "calculator.h"
#include "util/clUtil.h"


Calculator::Calculator(cl::Context& context,
                       const cl::Device& device,
                       int width, int height,
                       int maxNumKeypoints)
 :  commandQueue(context, device),
    dtcwt(context, {device}, commandQueue, 0.5f),
    abs(context, {device}),
    energyMap(context, {device}),
    peakDetector(context, {device}),
    descriptorExtracter_(context, {device}, peakDetector.getPosLength()),
    maxNumKeypoints_(maxNumKeypoints)
{
    const int numLevels = 3;
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


    descriptors_ = cl::Buffer(context, CL_MEM_READ_WRITE,
            maxNumKeypoints * 
              descriptorExtracter_.getNumFloatsInDescriptor() * sizeof(float));


    // Create the temporaries and results for peak detection
    peakDetectorResults = peakDetector.createResultsStructure
        (std::vector<size_t>(energyMaps.size(), maxNumKeypoints), 
         maxNumKeypoints);
    // i.e. allow the maximum number to appear in any given level, but
    // cap overall too to prevent getting more than we can store.
    
    descriptorsDone_ = std::vector<cl::Event>(energyMaps.size() * 2);
    // Enough for coarse and fine parts of descriptors being done

    // Set up the scales (used in peak detection)
    float s = 4.0f;
    for (int n = 0; n < energyMaps.size(); ++n) {
        scales.push_back(s);
        s *= 2.0f;
    }


}


#include <iostream>

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
    peakDetector(commandQueue, emPointers, scales, 0.1, 0.0f,
                               peakDetectorResults,
                               energyMapsDone);

    // Extract the descriptors
    for (size_t l = 0; l < (energyMaps.size() - 1); ++l) {
        descriptorExtracter_(commandQueue, 
                dtcwtOut.subbands[l], scales[l],      // Subband
                dtcwtOut.subbands[l+1], scales[l+1],  // Parent subband
                peakDetectorResults.list,         // Locations of keypoints
                peakDetectorResults.cumCounts, l, maxNumKeypoints_, 
                        // Start indices within list of the different 
                        // levels; which level to extract; what the maximum
                        // number of keypoints we could be asking for is.
                descriptors_,
                {peakDetectorResults.cumCountsDone},
                        // The cumulative counts rely on everything else
                        // in the peak detector being done
                &descriptorsDone_[l], &descriptorsDone_[l + energyMaps.size()]
                        // Wait for both coarse and fine to be done
                );
    }
    



#if 0
    // Display keypoint locations
    commandQueue.finish();
    // Read the last accumulated value for the total number of peaks
        int numOutputsVal;
        commandQueue.enqueueReadBuffer(peakDetectorResults.cumCounts, CL_TRUE, 
                             peakDetectorResults.levelLists.size() * sizeof(cl_uint), 
                             sizeof(cl_uint),
                             &numOutputsVal);



    // Now read the peaks themselves out
        std::vector<float> outputs(numOutputsVal 
                                    * peakDetectorResults.numFloatsPerPosition);
        commandQueue.enqueueReadBuffer(peakDetectorResults.list, CL_TRUE, 
                             0, outputs.size() * sizeof(float), 
                             &outputs[0]);

        std::cout << numOutputsVal << " outputs" << std::endl;

        // Display all the keypoints found: (x, y, scale, -)
        for (int n = 0; n < numOutputsVal; ++n) {

            for (int m = 0; m < peakDetectorResults.numFloatsPerPosition; ++m)
                std::cout << outputs[n * peakDetectorResults.numFloatsPerPosition + m] 
                          << "\t";
            std::cout << "\n";

        }
#endif


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


size_t Calculator::numFloatsPerKPLocation(void)
{
    return peakDetectorResults.numFloatsPerPosition;
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




