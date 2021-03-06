// Copyright (C) 2013 Timothy Gale
#ifndef ENERGY_MAP_BTK_H
#define ENERGY_MAP_BTK_H

#include "DTCWT/dtcwt.h"


class EnergyMapBTK {
    // The min operator over the subbands

public:

    EnergyMapBTK() = default;
    EnergyMapBTK(const EnergyMapBTK&) = default;

    EnergyMapBTK(cl::Context& context,
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

