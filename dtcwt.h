#ifndef DTCWT_H
#define DTCWT_H

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "cl.hpp"

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
    cl::Image2D xlo, lolo, lox, lohi, hilo, xbp, bpbp;
    cl::Event xloDone, loloDone, loxDone, lohiDone, hiloDone, xbpDone, bpbpDone;
};


struct Subbands {
    // 2-element images
    std::array<cl::Image2D, 6> sb;

    // List of events: when all done, all of sb are ready to use
    std::vector<cl::Event> done;
};


struct DtcwtEnv {
    size_t width, height;
    int numLevels, startLevel;

    std::vector<LevelTemps> levelTemps;

    cl::Context context_;
};


struct SubbandOutputs {

    SubbandOutputs(const DtcwtEnv& env);

    std::vector<Subbands> subbands;

};



class Dtcwt {
private:

    Filter h0x, h0y, h1x, h1y, hbpx, hbpy;
    DecimateFilter g0x, g0y, g1x, g1y, gbpx, gbpy;
    cl::Context context_;

    QuadToComplex quadToComplex;

    void filter(cl::CommandQueue& commandQueue,
                cl::Image2D& xx, const std::vector<cl::Event>& xxEvents,
                LevelTemps& levelTemps, Subbands* subbands);

    void decimateFilter(cl::CommandQueue& commandQueue,
                        cl::Image2D& xx, 
                        const std::vector<cl::Event>& xxEvents,
                        LevelTemps& levelTemps, Subbands* subbands);
public:

    Dtcwt(cl::Context& context, const std::vector<cl::Device>& devices,
          Filters level1, Filters leveln);

    void operator() (cl::CommandQueue& commandQueue,
                     cl::Image2D& image, 
                     DtcwtEnv& env,
                     SubbandOutputs& subbandOutputs);

    // Create the set of images etc needed to perform a DTCWT calculation
    DtcwtEnv createContext(size_t imageWidth, size_t imageHeight, 
                           size_t numLevels, size_t startLevel);

};

#endif
