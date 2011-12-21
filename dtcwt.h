#ifndef DTCWT_H
#define DTCWT_H

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "cl.hpp"

#include "filterer.h"

#include <vector>

class Dtcwt {
private:

    ColFilter colFilter;
    RowFilter rowFilter;
    ColDecimateFilter colDecimateFilter;
    RowDecimateFilter rowDecimateFilter;
    QuadToComplex quadToComplex;

    cl::Image2D colPadAndFilter(cl::Image2D&);
    cl::Image2D colPadAndDecFilter(cl::Image2D&);
    cl::Image2D rowPadAndFilter(cl::Image2D&);
    cl::Image2D rowPadAndDecFilter(cl::Image2D&);
    
public:

    Dtcwt(cl::Context& context, const std::vector<cl::Device>& devices);

    std::vector<std::vector<cl::Image2D> >
        operator() (cl::Image2D& image);
};

#endif
