// Copyright (C) 2013 Timothy Gale
#ifndef INT_DTCWT_H
#define INT_DTCWT_H

// Interleaved DTCWT implementation.  Uses the single tree DTCWT.

#include "dtcwt.h"
#include "Filter/imageBuffer.h"
#include "Filter/ScaleImageToImageBuffer/scaleImageToImageBuffer.h"

#include <tuple>
#include <vector>


// Forward declaration
class IntDtcwtOutput;

class IntDtcwt {

    // Class to perform an interleaved DTCWT operation

public:
    IntDtcwt() = default;
    IntDtcwt(const IntDtcwt&) = default;
    // Allow default and copy construction.  Nothing special for these,
    // so just allow it.

    IntDtcwt(cl::Context& context,
             std::vector<cl::Device>& devices,
             float scaleFactor = 1.f);
    // Scale factor specifies a constant to multiply by at each level.
    // Consider using 0.5 to keep magnitudes reasonably similar between
    // levels.
    
    void operator() (cl::CommandQueue& commandQueue,
                     cl::Image2D& image,
                     IntDtcwtOutput& output,
                     const std::vector<cl::Event>& waitEvents = {});
    // Analyse image into output, using the specified command queue after
    // waitEvents (if any) have completed.
    
    IntDtcwtOutput createOutputs(size_t width, size_t height, 
                                 size_t startLevel, size_t numLevels,
                                 const std::vector<float> &scales);
    // Create the outputs (including temporary variables)
                  

private:

    cl::Context context_;
    ScaleImageToImageBuffer scaleImageToImageBuffer_;
    Dtcwt dtcwt_;

};




class IntDtcwtOutput {

    // Class to hold the results of an interleaved DTCWT operation

    friend class IntDtcwt;
    // Allow IntDtcwt to manipulate the output behind the scenes

public:

    // Get the number of trees
    size_t numTrees() const;

    // Get the number of levels calculated per tree
    size_t numLevels() const;

    // Get the number of the first level calculated
    // in all the trees
    size_t startLevel() const;

    // Get the index of the output
    size_t idxFromTreeLevel(size_t tree, size_t level) const;

    // Convert from index to level and tree number
    std::tuple<size_t,size_t> treeLevelFromIdx(size_t idx) const;

    // Get the scale of the level number or index
    float scale(size_t tree, size_t level) const;
    float scale(size_t idx) const;

    // Get requested subbands by tree and tree level
    Subbands& level(size_t tree, size_t level);
    const Subbands& level(size_t tree, size_t level) const;

    // Get the subband by index size_to all the trees
    Subbands& operator[](size_t idx);
    const Subbands& operator[](size_t idx) const;

    // Get the events that will be true when completed by tree & level
    std::vector<cl::Event> doneEvents(size_t tree, size_t level) const;

    // Get the events that will be true when completed by index
    std::vector<cl::Event> doneEvents(size_t idx) const;

private:

    float inputWidth_, inputHeight_;
    size_t startLevel_, numLevels_;

    std::vector<float> scales_;

    std::vector<ImageBuffer<cl_float>> inputImages_;
    std::vector<DtcwtTemps> dtcwtTemps_;
    std::vector<DtcwtOutput> dtcwtOutputs_;

};

#endif

