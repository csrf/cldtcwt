// Copyright (C) 2013 Timothy Gale
#ifndef ENERGY_MAP_H
#define ENERGY_MAP_H

#include "DTCWT/dtcwt.h"


class EnergyMap {
    // Class that converts an interleaved image to two subbands with real
    // and imaginary components

public:

    EnergyMap() = default;
    EnergyMap(const EnergyMap&) = default;

    EnergyMap(cl::Context& context,
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

