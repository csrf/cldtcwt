#include "dtcwt.h"

Dtcwt::Dtcwt(cl::Context& context, const std::vector<cl::Device>& devices)
    : colFilter(context, devices),
      rowFilter(context, devices),
      colDecimateFilter(context, devices),
      rowDecimateFilter(context, devices),
      quadToComplex(context, devices)
{}


std::vector<std::vector<cl::Image2D> >
    Dtcwt::operator() (cl::Image2D& image)
{
    std::vector<std::vector<cl::Image2D> > result;

    return result;
}


cl::Image2D Dtcwt::colPadAndFilter(cl::Image2D& image)
{
}


cl::Image2D Dtcwt::colPadAndDecFilter(cl::Image2D&)
{}


cl::Image2D Dtcwt::rowPadAndFilter(cl::Image2D&)
{}


cl::Image2D Dtcwt::rowPadAndDecFilter(cl::Image2D&)
{}


