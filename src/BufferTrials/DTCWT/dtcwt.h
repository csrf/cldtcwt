#ifndef DTCWT_H
#define DTCWT_H

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"

#include "../imageBuffer.h"

#include "../PadX/padX.h"
#include "../PadY/padY.h"

#include "../FilterX/filterX.h"
#include "../FilterY/filterY.h"
#include "../QuadToComplex/quadToComplex.h"

#include "../DecimateFilterX/decimateFilterX.h"
#include "../DecimateTripleFilterX/decimateTripleFilterX.h"
#include "../DecimateFilterY/decimateFilterY.h"
#include "../QuadToComplexDecimateFilterY/q2cDecimateFilterY.h"

#include <vector>
#include <tuple>
#include <array>


// Temporary images needed only when the level produces an output 
struct LevelTemps {

    // Rows filtered
    ImageBuffer lo, hi, bp;

    // Columns & rows filtered for next stage
    ImageBuffer lolo;

    // These only get used if producing outputs at Level 1
    ImageBuffer lohi, hilo, bpbp;
    cl::Event lohiDone, hiloDone, bpbpDone;
    
    // Done events for each of these
    cl::Event loDone, hiDone, bpDone, 
              loloDone; 
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

    PadX padX;
    PadY padY;
    
    FilterX h0ox, h1ox, h2ox;
    FilterY h0oy, h1oy, h2oy;

    QuadToComplex quadToComplex;

    DecimateFilterX h0bx;
    DecimateFilterY h0by;

    DecimateTripleFilterX h012bx;

    QuadToComplexDecimateFilterY q2ch0by, q2ch1by, q2ch2by;

    const size_t padding_ = 16;
    const size_t alignment_ = 32;

// Debug:
public:
    void filter(cl::CommandQueue& commandQueue,
                ImageBuffer& xx, const std::vector<cl::Event>& xxEvents,
                LevelTemps& levelTemps, LevelOutput* subbands);

    void decimateFilter(cl::CommandQueue& commandQueue,
                        ImageBuffer& xx, 
                        const std::vector<cl::Event>& xxEvents,
                        LevelTemps& levelTemps, LevelOutput* subbands);
public:

    Dtcwt() = default;
    Dtcwt(const Dtcwt&) = default;

    Dtcwt(cl::Context& context, const std::vector<cl::Device>& devices,
          float scaleFactor = 1.f);
    // Scale factor selects how much to multiply each level by,
    // cumulatively.  0.5 is useful in quite a few cases, because otherwise
    // the coarser scales have much greater magnitudes.

    void operator() (cl::CommandQueue& commandQueue,
                     ImageBuffer& image, 
                     DtcwtTemps& env,
                     DtcwtOutput& subbandOutputs,
                     const std::vector<cl::Event>& waitEvents
                        = std::vector<cl::Event>());

    // Create the set of images etc needed to perform a DTCWT calculation
    DtcwtTemps createContext(size_t imageWidth, size_t imageHeight, 
                           size_t numLevels, size_t startLevel);

};



#endif

