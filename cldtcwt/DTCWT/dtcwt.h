// Copyright (C) 2013 Timothy Gale
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
#include "Filter/TripleQuadToComplexDecimateFilterY/tripleQ2cDecimateFilterY.h"

#include <vector>
#include <tuple>
#include <array>

// Forward declaration, so the processor can be used as a friend
class Dtcwt;
class DtcwtTemps;
class DtcwtOutput;

// Temporary images used in the production of an output level
struct LevelTemps {

    LevelTemps();
    LevelTemps(cl::Context& context,
               size_t inputWidth, size_t inputHeight,
               size_t padding, size_t alignment,
               bool isLevelOne,
               bool producesOutputs);

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

    bool isLevelOne_, producesOutputs_;
    size_t inputWidth_, inputHeight_;
    size_t outputWidth_, outputHeight_;

};



class DtcwtTemps {

    friend class Dtcwt;

private:
    cl::Context context_;

    size_t width_, height_;
    int numLevels_, startLevel_;

    size_t padding_= 16,
           alignment_ = 32;

    std::vector<LevelTemps> levelTemps_;

public:
    DtcwtOutput createOutputs();

    DtcwtTemps(cl::Context& context,
               size_t imageWidth, size_t imageHeight, 
               size_t startLevel, size_t numLevels);
    DtcwtTemps() = default;
};



typedef ImageBuffer<Complex<cl_float>> Subbands;


class DtcwtOutput {

    // Constructed by
    friend class DtcwtTemps;

    // Modified by
    friend class Dtcwt;

private:
    std::vector<Subbands> levels_;
    std::vector<std::vector<cl::Event>> doneEvents_;

    size_t startLevel_;
    size_t numLevels_;

public:


    // Return the specified level (1 is the first level of the tree,
    // etc)
    Subbands& level(int levelNum);
    const Subbands& level(int levelNum) const;

    // Return the output level (0 is the first level producing a level,
    // etc)
    Subbands& operator [] (int n);
    const Subbands& operator [] (int n) const;


    // begin and end allow us to iterator over the levels using for
    std::vector<Subbands>::iterator begin();
    std::vector<Subbands>::const_iterator begin() const;
    std::vector<Subbands>::iterator end();
    std::vector<Subbands>::const_iterator end() const;


    std::vector<cl::Event> doneEvents(int levelNum);
    const std::vector<cl::Event> doneEvents(int levelNum) const;

    size_t startLevel() const;
    size_t numLevels() const;

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

    DecimateTripleFilterX h021bx;

    TripleQuadToComplexDecimateFilterY q2c_h1_h2_h0;

    const size_t padding_ = 16;
    const size_t alignment_ = 32;

// Debug:
public:
    void filter(cl::CommandQueue& commandQueue,
                ImageBuffer<cl_float>& xx, 
                const std::vector<cl::Event>& xxEvents,
                LevelTemps& levelTemps, 
                Subbands* subbands,
                std::vector<cl::Event>* events);

    void decimateFilter(cl::CommandQueue& commandQueue,
                        ImageBuffer<cl_float>& xx, 
                        const std::vector<cl::Event>& xxEvents,
                        LevelTemps& levelTemps, 
                        Subbands* subbands,
                        std::vector<cl::Event>* events);

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

    /// @brief Member function based interface to ::operator().
    ///
    /// @param commandQueue
    /// @param image
    /// @param env
    /// @param subbandOutputs
    /// @param 
    void transform (cl::CommandQueue& commandQueue,
                     ImageBuffer<cl_float>& image, 
                     DtcwtTemps& env,
                     DtcwtOutput& subbandOutputs,
                     const std::vector<cl::Event>& waitEvents
                        = std::vector<cl::Event>())
    {
        (*this)(commandQueue, image, env, subbandOutputs, waitEvents);
    }

};



#endif


