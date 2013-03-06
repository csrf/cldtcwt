#ifndef INTERPMAPEIGEN_H
#define INTERPMAPEIGEN_H

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"

#include "DTCWT/dtcwt.h"

class InterpMapEigen {
    // Class that calculates an energy map using an estimate of how quickly
    // the subband values change on motion in the least-sensitive direction.
    
public:

    InterpMapEigen() = default;
    InterpMapEigen(const InterpMapEigen&) = default;

    InterpMapEigen(cl::Context& context,
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


