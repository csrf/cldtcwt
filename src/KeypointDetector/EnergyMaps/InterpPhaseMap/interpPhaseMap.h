#ifndef INTERP_PHASE_MAP_H
#define INTERP_PHASE_MAP_H

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"

#include "DTCWT/dtcwt.h"

class InterpPhaseMap {
    // Class that calculates an energy map using an estimate of how quickly
    // the subband values change on motion in the least-sensitive direction.
    // This version is different to InterpMapEigen in that it interpolates
    // the phase and magitude of the coefficients separately, so that the
    // underlying frequency of the signal is respected.  The previous way
    // used the bandpass method, which works well, but as implemented
    // seemed to have issues if the actual frequency differed too much from 
    // the subband's centre frequency (e.g. because it was actually leakage
    // from another subband, but too big just to ignore).
    
public:

    InterpPhaseMap() = default;
    InterpPhaseMap(const InterpPhaseMap&) = default;

    InterpPhaseMap(cl::Context& context,
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


