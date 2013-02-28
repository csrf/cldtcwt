#include "peakDetector.h"
#include <stdexcept>


PeakDetector::PeakDetector(cl::Context& context,
                           const std::vector<cl::Device>& devices)
 : context_(context),
   findMax_(context, devices),
   accumulate_(context, devices),
   concat_(context, devices)
{
    float zerof = 0.0f;

    // Create a zero image to let the max find work at the edges
    zeroImage_ = cl::Image2D(context, 
                             CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                             cl::ImageFormat(CL_LUMINANCE, CL_FLOAT),
                             1, 1, 0,
                             &zerof);
}



PeakDetectorResults PeakDetector::createResultsStructure
        (const std::vector<size_t>& maxLevelCounts, size_t maxTotalCount)
{
    // Create the intermediates and final outputs.
    PeakDetectorResults results;

    results.numFloatsPerPosition = findMax_.getPosLength();

    // Per-level counts
    results.counts = cl::Buffer(context_, CL_MEM_READ_WRITE,
                                maxLevelCounts.size() * sizeof(cl_uint));

    // For use when zeroing the counts: we need memory that will persist
    // even when not in a particular function, and which is in no danger
    // of going away while doing an operation.
    results.zeroCounts = std::vector<cl_uint>(maxLevelCounts.size(), 0);

    // Per-level lists
    for (size_t maxCount: maxLevelCounts) 
        // Allocate space to put the results into, and somewhere to store
        // their done events
        results.levelLists.emplace_back(context_, CL_MEM_READ_WRITE,
                    maxCount * results.numFloatsPerPosition * sizeof(float));

    results.maxLevelCounts = maxLevelCounts;
    results.levelListsDone.resize(maxLevelCounts.size());

    // Cumulative counts
    results.cumCounts = cl::Buffer(context_, CL_MEM_READ_WRITE,
                               (maxLevelCounts.size() + 1) * sizeof(cl_uint));

    // Concatenated list
    results.list = cl::Buffer(context_, CL_MEM_READ_WRITE,
                maxTotalCount * results.numFloatsPerPosition * sizeof(float));

    results.maxListLength = maxTotalCount;

    results.listDone.resize(maxLevelCounts.size());

    return results;
}



void PeakDetector::operator() (cl::CommandQueue& cq,
                               const std::vector<cl::Image*> energyMaps,
                               const std::vector<float> scales,
                               float threshold, float eigenRatioThreshold,
                               PeakDetectorResults& results,
                               const std::vector<cl::Event>& waitEvents)
{
    // Check we have been given the right number of scales and energyMaps
    if (energyMaps.size() != results.levelLists.size())
        throw std::logic_error("PeakDetector: wrong number of energy maps");

    if (scales.size() != results.levelLists.size())
        throw std::logic_error("PeakDetector: wrong number of scales");

    // Clear the counts
    cq.enqueueWriteBuffer(results.counts, CL_FALSE, 
                          0, results.zeroCounts.size() * sizeof(cl_uint), 
                          &results.zeroCounts[0],
                          nullptr, &results.countsCleared);
    

    // Find the maxima (which means clearing needs to have finished)
    std::vector<cl::Event> findWaitEvents = waitEvents;
    findWaitEvents.push_back(results.countsCleared);

    for (int n = 0; n < energyMaps.size(); ++n) {

        // Work out what the finer image is (zero if none)
        cl::Image* finerImage = &zeroImage_;
        float finerScale = 1.f;
        if (n > 0) {
            finerImage = energyMaps[n-1];
            finerScale = scales[n-1];
        }

        // Work out what the coarser image is (zero if none)
        cl::Image* coarserImage = &zeroImage_;
        float coarserScale = 1.f;
        if (n < (energyMaps.size() - 1)) {
            coarserImage = energyMaps[n+1];
            coarserScale = scales[n+1];
        } 

        // Execute the kernel
        findMax_(cq, *energyMaps[n], scales[n],
                     *finerImage, finerScale,
                     *coarserImage, coarserScale,
                     threshold, eigenRatioThreshold,
                     results.levelLists[n], 
                     results.counts, n,
                     findWaitEvents, &results.levelListsDone[n]);
    }

    // Accumulate the counts
    accumulate_(cq, results.counts, results.cumCounts,
                    results.maxListLength,
                    results.levelListsDone, 
                    &results.cumCountsDone);

    // Concatenate the maximum positions
    for (int n = 0; n < results.levelLists.size(); ++n) 
        concat_(cq, results.levelLists[n], results.list,
                    results.cumCounts, n,
                    results.numFloatsPerPosition,
                    {results.cumCountsDone},
                    &results.listDone[n]);

}



size_t PeakDetector::getPosLength()
{
    return findMax_.getPosLength();
}

