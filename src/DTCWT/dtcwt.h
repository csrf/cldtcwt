#ifndef DTCWT_H
#define DTCWT_H

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "CL/cl.hpp"

#include "Filter/imageBuffer.h"

#include "Filter/PadX/padX.h"
#include "Filter/PadY/padY.h"

#include "Filter/FilterX/filterX.h"
#include "Filter/FilterY/filterY.h"
#include "Filter/QuadToComplex/quadToComplex.h"

#include "Filter/DecimateFilterX/decimateFilterX.h"
#include "Filter/DecimateTripleFilterX/decimateTripleFilterX.h"
#include "Filter/DecimateFilterY/decimateFilterY.h"
#include "Filter/QuadToComplexDecimateFilterY/q2cDecimateFilterY.h"

#include <vector>
#include <tuple>
#include <array>


// Temporary images needed only when the level produces an output 
struct LevelTemps {

    // Rows filtered
    ImageBuffer<cl_float> lo, hi, bp;

    // Columns & rows filtered for next stage
    ImageBuffer<cl_float> lolo;

    // These only get used if producing outputs at Level 1
    ImageBuffer<cl_float> lohi, hilo, bpbp;
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

    // Subband matrices
    std::array<ImageBuffer<Complex<cl_float>>, 6> sb;

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
                ImageBuffer<cl_float>& xx, 
                const std::vector<cl::Event>& xxEvents,
                LevelTemps& levelTemps, LevelOutput* subbands);

    void decimateFilter(cl::CommandQueue& commandQueue,
                        ImageBuffer<cl_float>& xx, 
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
                     ImageBuffer<cl_float>& image, 
                     DtcwtTemps& env,
                     DtcwtOutput& subbandOutputs,
                     const std::vector<cl::Event>& waitEvents
                        = std::vector<cl::Event>());

    // Create the set of images etc needed to perform a DTCWT calculation
    DtcwtTemps createContext(size_t imageWidth, size_t imageHeight, 
                           size_t numLevels, size_t startLevel);

};



#endif


