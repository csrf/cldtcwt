#ifndef PEAKDETECTOR_H
#define PEAKDETECTOR_H

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"
#include <vector>

#include "Concat/concat.h"
#include "FindMax/findMax.h"
#include "Accumulate/accumulate.h"

class PeakDetector;


struct PeakDetectorResults {

private:
    // Number of floats used for each position detected
    size_t numFloatsPerPosition_;

    // Intermediates: the per-level lists (as opposed to the full one)
    std::vector<cl_uint> zeroCounts_; // For zeroing the counts
    cl::Buffer counts_;
    cl::Event countsCleared_;
    std::vector<cl::Buffer> levelLists_;
    std::vector<size_t> maxLevelCounts_;
    std::vector<cl::Event> levelListsDone_;

    // Counts from each level
    cl::Buffer cumCounts_;
    cl::Event cumCountsDone_;

    // List of positions relative to the image centre with scales
    cl::Buffer list_;
    std::vector<cl::Event> listDone_;
    size_t maxListLength_;

public:
    size_t numFloatsPerPosition() const;
    // Number of floats per position
    
    size_t numLevels() const;
    // Number of levels analysed

    cl::Buffer cumCounts() const;
    cl::Event cumCountsDone() const;
    // Array of cumulative counts, starting with zero

    cl::Buffer list() const;
    std::vector<cl::Event> listDone() const;
    // List of peak locations.

    friend PeakDetector;

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

