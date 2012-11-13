#ifndef EXTRACT_DESCRIPTORS_H
#define EXTRACT_DESCRIPTORS_H

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"
#include <vector>


#include "DTCWT/dtcwt.h"


struct Coord {
    float x, y;
};

class Interpolator {
// Class that can extract a ring of keypoints from a particular level

public:

    Interpolator() = default;
    Interpolator(const Interpolator&) = default;

    Interpolator(cl::Context& context,
                        const std::vector<cl::Device>& devices,
                        std::vector<Coord> samplingPattern,
                        int outputStride, int outputOffset,
                        int diameter,
                        int numFloatsPerPos);
    // numFloatsPerPos - The number of floating points taken to describe
    // each position. The first two of these are x and y relative to the
    // centre of the image at the untransformed image scale.

    void
    operator() (cl::CommandQueue& cq,
                const LevelOutput& subbands,
                const cl::Buffer& locations,
                float scale,
                const cl::Buffer& kpOffsets,
                int kpOffsetsIdx,
                int maxNumKPs,
                cl::Buffer& output,
                std::vector<cl::Event> waitEvents = std::vector<cl::Event>(),
                cl::Event* doneEvent = nullptr);
    // scale - the number of original image pixels per pixel at this level 
    // of transform.


private:

    cl::Context context_;
    cl::Kernel kernel_;

    cl::Buffer samplingPattern_;
    int diameter_;

};


class DescriptorExtracter {
// Extract descriptors from two consecutive levels, the lower one a ring
// with a central point (unit radius) and the upper one a circle.  The
// coordinates are for the finer scale and relative to its centre

public:
    DescriptorExtracter() = default;
    DescriptorExtracter(const DescriptorExtracter&) = default;

    DescriptorExtracter(cl::Context& context, 
                        const std::vector<cl::Device>& devices,
                        int numFloatsPerPos);

    void
    operator() (cl::CommandQueue& cq,
                const LevelOutput& fineSubbands,   
                float fineScale,
                const LevelOutput& coarseSubbands,
                float coarseScale,
                const cl::Buffer& locations,
                const cl::Buffer& kpOffsets,
                int kpOffsetsIdx,
                int maxNumKPs,
                cl::Buffer& output,
                std::vector<cl::Event> waitEvents = std::vector<cl::Event>(),
                cl::Event* doneEventFine = nullptr,
                cl::Event* doneEventCoarse = nullptr);


private:

    // Different interpolators for the ring vs. the single point
    Interpolator fineInterpolator_, coarseInterpolator_;

};




#endif

