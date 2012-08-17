#ifndef DTCWT_H
#define DTCWT_H

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"

#include "filterer.h"

#include <vector>
#include <tuple>
#include <array>

struct Filters {
    // Low pass, high pass and band pass coefficients (respectively)
    cl::Buffer h0, h1, hbp;
};


// Temporary images needed only when the level produces an output 
struct LevelTemps {

    // Columns filtered
    cl::Image2D lo, bp, hi;

    // Columns & rows filtered
    cl::Image2D lolo, lohi, hilo, bpbp;

    // Done events for each of these
    cl::Event loDone, bpDone, hiDone,
              loloDone, lohiDone, hiloDone, bpbpDone;

};



struct DtcwtTemps {
    size_t width, height;
    int numLevels, startLevel;

    std::vector<LevelTemps> levelTemps;

    cl::Context context_;
};



struct LevelOutput {
    // 2-element images
    std::array<cl::Image2D, 6> sb;

    // List of events: when all done, all of sb are ready to use
    std::vector<cl::Event> done;
};



struct DtcwtOutput {

    DtcwtOutput(const DtcwtOutput&) = default;
    DtcwtOutput() = default;
    DtcwtOutput(const DtcwtTemps& env);

    std::vector<LevelOutput> subbands;

};



class Dtcwt {
private:

    cl::Context context_;

    Filters level1_, leveln_;

    Filter h0x, h0y, h1x, h1y, hbpx, hbpy;
    DecimateFilter g0x, g0y, g1x, g1y, gbpx, gbpy;


    QuadToComplex quadToComplex;

// Debug:
public:
    void filter(cl::CommandQueue& commandQueue,
                cl::Image& xx, const std::vector<cl::Event>& xxEvents,
                LevelTemps& levelTemps, LevelOutput* subbands);

    void decimateFilter(cl::CommandQueue& commandQueue,
                        cl::Image2D& xx, 
                        const std::vector<cl::Event>& xxEvents,
                        LevelTemps& levelTemps, LevelOutput* subbands);
public:

    Dtcwt() = default;
    Dtcwt(const Dtcwt&) = default;

    Dtcwt(cl::Context& context, const std::vector<cl::Device>& devices,
          cl::CommandQueue commandQueue, float scaleFactor = 1.f);
    // Scale factor selects how much to multiply each level by,
    // cumulatively.  0.5 is useful in quite a few cases, because otherwise
    // the coarser scales have much greater magnitudes.

    void operator() (cl::CommandQueue& commandQueue,
                     cl::Image& image, 
                     DtcwtTemps& env,
                     DtcwtOutput& subbandOutputs,
                     const std::vector<cl::Event>& waitEvents
                        = std::vector<cl::Event>());

    // Create the set of images etc needed to perform a DTCWT calculation
    DtcwtTemps createContext(size_t imageWidth, size_t imageHeight, 
                           size_t numLevels, size_t startLevel);

};




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
           const LevelOutput& levelOutput,
           cl::Image2D& energyMap,
           cl::Event* doneEvent = nullptr);

private:
    cl::Context context_;
    cl::Kernel kernel_;

};

#endif


