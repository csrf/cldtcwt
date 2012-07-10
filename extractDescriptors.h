#ifndef EXTRACT_DESCRIPTORS_H
#define EXTRACT_DESCRIPTORS_H

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "cl.hpp"
#include <vector>





class DescriptorExtracter {
// Class that can extract a ring of keypoints from a particular level

public:

    DescriptorExtracter() = default;
    DescriptorExtracter(const DescriptorExtracter&) = default;

    DescriptorExtracter(cl::Context& context,
                        const std::vector<cl::Device>& devices);



private:

    cl::Context context_;
    cl::Kernel kernel_;

};





#endif

