#ifndef ENERGYMAPEIGEN_H
#define ENERGYMAPEIGEN_H

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"

#include "DTCWT/dtcwt.h"

class EnergyMapEigen {
    // Class that converts an interleaved image to two subbands with real
    // and imaginary components

public:

    EnergyMapEigen() = default;
    EnergyMapEigen(const EnergyMapEigen&) = default;

    EnergyMapEigen(cl::Context& context,
              const std::vector<cl::Device>& devices);

    void
    operator() (cl::CommandQueue& commandQueue,
           const Subbands& levelOutput,
           cl::Image2D& energyMap,
           const std::vector<cl::Event>& preconditions = {},
           cl::Event* doneEvent = nullptr);

private:
    cl::Context context_;
    cl::Kernel kernel_;

};

#endif


