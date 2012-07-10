#ifndef EXTRACT_DESCRIPTORS_H
#define EXTRACT_DESCRIPTORS_H

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "cl.hpp"
#include <vector>


#include "dtcwt.h"



class DescriptorExtracter {
// Class that can extract a ring of keypoints from a particular level

public:

    DescriptorExtracter() = default;
    DescriptorExtracter(const DescriptorExtracter&) = default;

    DescriptorExtracter(cl::Context& context,
                        const std::vector<cl::Device>& devices,
                        cl::CommandQueue& cq,
                        const std::vector<float[2]>& samplingPattern,
                        float scaleFactor,
                        int outputStride, int outputOffset,
                        int diameter);

    void
    operator() (cl::CommandQueue& cq,
                const LevelOutput& subbands,
                const cl::Buffer& locations,
                int numLocations,
                cl::Buffer& output,
                cl::Event* doneEvent = nullptr);


private:

    cl::Context context_;
    cl::Kernel kernel_;

    cl::Buffer samplingPattern_;
    int diameter_;

};





#endif

