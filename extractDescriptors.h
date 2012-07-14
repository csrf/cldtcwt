#ifndef EXTRACT_DESCRIPTORS_H
#define EXTRACT_DESCRIPTORS_H

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "cl.hpp"
#include <vector>


#include "dtcwt.h"


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
                        cl::CommandQueue& cq,
                        const std::vector<Coord>& samplingPattern,
                        float scaleFactor,
                        int outputStride, int outputOffset,
                        int diameter);

    void
    operator() (cl::CommandQueue& cq,
                const LevelOutput& subbands,
                const cl::Buffer& locations,
                int numLocations,
                cl::Buffer& output,
                std::vector<cl::Event> waitEvents = std::vector<cl::Event>(),
                cl::Event* doneEvent = nullptr);


private:

    cl::Context context_;
    cl::Kernel kernel_;

    cl::Buffer samplingPattern_;
    int diameter_;

};





#endif

