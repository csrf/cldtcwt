#ifndef PEAKDETECTOR_H
#define PEAKDETECTOR_H

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"
#include <vector>

#include "KeypointDetector/concat.h"
#include "KeypointDetector/findMax.h"
#include "MiscKernels/accumulate.h"

struct PeakDetectorResults {

    // Number of floats used for each position detected
    size_t numFloatsPerPosition;

    // Intermediates: the per-level lists (as opposed to the full one)
    std::vector<cl_uint> zeroCounts; // For zeroing the counts
    cl::Buffer counts;
    cl::Event countsCleared;
    std::vector<cl::Buffer> levelLists;
    std::vector<size_t> maxLevelCounts;
    std::vector<cl::Event> levelListsDone;

    // Counts from each level
    cl::Buffer cumCounts;
    cl::Event cumCountsDone;

    // List of positions relative to the image centre with scales
    cl::Buffer list;
    std::vector<cl::Event> listDone;
    size_t maxListLength;

};


class PeakDetector {

    // Takes a list of energy maps and scale values, and finds where the 
    // peaks are, putting them into a list.  A second list says where
    // each scale starts within that list.

private:

    cl::Context context_;

    cl::Image2D zeroImage_;

    // Kernels to use
    FindMax findMax_;
    Accumulate accumulate_;
    Concat concat_;

public:

    PeakDetector() = default;
    PeakDetector(const PeakDetector&) = default;
    PeakDetector(cl::Context& context,
                 const std::vector<cl::Device>& devices);

    PeakDetectorResults createResultsStructure
        (const std::vector<size_t>& maxLevelCounts,
         size_t maxTotalCount);

    void operator() (cl::CommandQueue& cq,
                     const std::vector<cl::Image*> energyMaps,
                     const std::vector<float> scales, // Scales of the corresponding
                                                      // maps
                     float threshold, // Minimum peak height to detect
                     float eigenRatioThreshold, // Minimum ratio between the 
                                    // eigenvalues of the curvature, to 
                                    // try to supress edges
                     PeakDetectorResults& results,
                     const std::vector<cl::Event>& waitEvents = {});

    size_t getPosLength();
    // Returns the number of floats in the position vector

};

#endif

