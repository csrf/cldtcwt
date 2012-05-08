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


// Temporary images used whether the level produces an output or not
struct NoOutputTemps {
    cl::Image2D xlo, lolo;
};


// Temporary images needed only when the level produces an output 
struct OutputTemps {
    cl::Image2D lox, lohi, hilo, xbp, bpbp;
};

typedef std::array<cl::Image2D, 6> Subbands;

struct DtcwtContext {
    size_t width, height;
    int numLevels, startLevel;

    std::vector<NoOutputTemps> noOutputTemps;
    std::vector<OutputTemps>   outputTemps;

    Filters level1, level2;

    // Outputs
    std::vector<Subbands> outputs;
};



class Dtcwt {
private:

    Filter h0x, h0y, h1x, h1y, hbpx, hbpy;
    DecimateFilter g0x, g0y, g1x, g1y, gbpx, gbpy;

    ColFilter colFilter;
    RowFilter rowFilter;
    ColDecimateFilter colDecimateFilter;
    RowDecimateFilter rowDecimateFilter;
    QuadToComplex quadToComplex;

    std::tuple<std::vector<Subbands>,
               std::vector<OutputTemps>,
               std::vector<NoOutputTemps>>
        dummyRun(size_t width, size_t height, int numLevels, int startLevel);

    std::tuple<OutputTemps, Subbands>
        dummyFilter(size_t width, size_t height, cl::Image2D xlo);
    std::tuple<OutputTemps, Subbands>
        dummyFilter(cl::Image2D xx, cl::Image2D xlo);

    std::vector<cl::Event> 
        filter(cl::CommandQueue& commandQueue,
               cl::Image2D& xx, 
               const std::vector<cl::Event>& xxEvents,
               cl::Image2D& xlo, 
               const std::vector<cl::Event>& xloEvents,
               cl::Image2D* out, 
               OutputTemps* outputTemps,
               Filters& filters);

    std::tuple<OutputTemps, Subbands>
        dummyDecimateFilter(size_t width, size_t height, cl::Image2D xlo);
    std::tuple<OutputTemps, Subbands>
        dummyDecimateFilter(cl::Image2D xx, cl::Image2D xlo);

    std::vector<cl::Event>
        decimateFilter(cl::CommandQueue& commandQueue,
                       cl::Image2D& xx, 
                       const std::vector<cl::Event>& xxEvents,
                       cl::Image2D& xlo, 
                       const std::vector<cl::Event>& xloEvent,
                       cl::Image2D* out, 
                       OutputTemps* outputTemps,
                       Filters& filters);

public:

    Dtcwt(cl::Context& context, const std::vector<cl::Device>& devices,
          Filters level1, Filters leveln);

    void operator() (cl::CommandQueue& commandQueue,
                     cl::Image2D& image, 
                     DtcwtContext& env);

    // Create the set of images etc needed to perform a DTCWT calculation
    DtcwtContext createContext(size_t imageWidth, size_t imageHeight, 
                               size_t numLevels, size_t startLevel,
                               Filters level1, Filters level2);

};

#endif
