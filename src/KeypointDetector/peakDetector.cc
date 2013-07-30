// Copyright (C) 2013 Timothy Gale
#include "peakDetector.h"
#include <stdexcept>



size_t PeakDetectorResults::numFloatsPerPosition() const
{
    return numFloatsPerPosition_;
}


cl::Buffer PeakDetectorResults::cumCounts() const
{
    return cumCounts_;
}


cl::Event PeakDetectorResults::cumCountsDone() const
{
    return cumCountsDone_;
}



cl::Buffer PeakDetectorResults::list() const
{
    return list_;
}


std::vector<cl::Event> PeakDetectorResults::listDone() const
{
    return listDone_;
}


size_t PeakDetectorResults::numLevels() const
{
    return zeroCounts_.size();
}






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

    results.numFloatsPerPosition_ = findMax_.getPosLength();

    // Per-level counts
    results.counts_ = cl::Buffer(context_, CL_MEM_READ_WRITE,
                                maxLevelCounts.size() * sizeof(cl_uint));

    // For use when zeroing the counts: we need memory that will persist
    // even when not in a particular function, and which is in no danger
    // of going away while doing an operation.
    results.zeroCounts_ = std::vector<cl_uint>(maxLevelCounts.size(), 0);

    // Per-level lists
    for (size_t maxCount: maxLevelCounts) 
        // Allocate space to put the results into, and somewhere to store
        // their done events
        results.levelLists_.emplace_back(context_, CL_MEM_READ_WRITE,
                    maxCount * results.numFloatsPerPosition_ * sizeof(float));

    results.maxLevelCounts_ = maxLevelCounts;
    results.levelListsDone_.resize(maxLevelCounts.size());

    // Cumulative counts
    results.cumCounts_ = cl::Buffer(context_, CL_MEM_READ_WRITE,
                               (maxLevelCounts.size() + 1) * sizeof(cl_uint));

    // Concatenated list
    results.list_ = cl::Buffer(context_, CL_MEM_READ_WRITE,
                maxTotalCount * results.numFloatsPerPosition_ * sizeof(float));

    results.maxListLength_ = maxTotalCount;

    results.listDone_.resize(maxLevelCounts.size());

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
    if (energyMaps.size() < results.levelLists_.size())
        throw std::logic_error("PeakDetector: wrong number of energy maps");

    if (scales.size() < results.levelLists_.size())
        throw std::logic_error("PeakDetector: wrong number of scales");

    // Clear the counts
    cq.enqueueWriteBuffer(results.counts_, CL_FALSE, 
                          0, results.zeroCounts_.size() * sizeof(cl_uint), 
                          &results.zeroCounts_[0],
                          nullptr, &results.countsCleared_);
    

    // Find the maxima (which means clearing needs to have finished)
    std::vector<cl::Event> findWaitEvents = waitEvents;
    findWaitEvents.push_back(results.countsCleared_);

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
                     results.levelLists_[n], 
                     results.counts_, n,
                     findWaitEvents, &results.levelListsDone_[n]);
    }

    // Accumulate the counts
    accumulate_(cq, results.counts_, results.cumCounts_,
                    results.maxListLength_,
                    results.levelListsDone_, 
                    &results.cumCountsDone_);

    // Concatenate the maximum positions
    for (int n = 0; n < results.levelLists_.size(); ++n) 
        concat_(cq, results.levelLists_[n], results.list_,
                    results.cumCounts_, n,
                    results.numFloatsPerPosition_,
                    {results.cumCountsDone_},
                    &results.listDone_[n]);

}



size_t PeakDetector::getPosLength()
{
    return findMax_.getPosLength();
}

