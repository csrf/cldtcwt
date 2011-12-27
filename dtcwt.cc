#include "dtcwt.h"

Dtcwt::Dtcwt(cl::Context& context, const std::vector<cl::Device>& devices)
    : colFilter(context, devices),
      rowFilter(context, devices),
      colDecimateFilter(context, devices),
      rowDecimateFilter(context, devices),
      quadToComplex(context, devices)
{}


std::vector<std::vector<cl::Image2D> >
    Dtcwt::operator() (cl::CommandQueue& commandQueue,
                       cl::Image2D& image, 
                       Filters level1, Filters level2,
                       int numLevels, int startLevel)
{
    std::vector<std::vector<cl::Image2D> > result;

    cl::Event loEvent, loloEvent;

    cl::Image2D xlo = colFilter(commandQueue, image, level1.h0,
                                {}, &loEvent);

    if (startLevel == 0) {
        cl::Event loxEvent;
        cl::Image2D lox = rowFilter(commandQueue, image, level1.h0,
                                    {}, &loxEvent);
        cl::Image2D lohi = colFilter(commandQueue, lox, level1.h1,
                                     {loxEvent});

        cl::Image2D hilo = rowFilter(commandQueue, xlo, level1.h1,
                                     {loEvent});

        cl::Event xbpEvent;
        cl::Image2D xbp = colFilter(commandQueue, image, level1.hbp,
                                    {}, &xbpEvent);
        cl::Image2D bpbp = rowFilter(commandQueue, xbp, level1.hbp,
                                     {xbpEvent});


    }

    cl::Image2D lolo = rowFilter(commandQueue, xlo, level1.h0, 
                                 {loEvent}, &loloEvent);


    for (int l = 1; l < numLevels; ++l) {

        xlo = colDecimateFilter(commandQueue, lolo, level2.h0,
                                {loloEvent}, &loEvent);

        // High pass only when interested in the outcome
        if (l >= startLevel) {
            cl::Event loxEvent;
            cl::Image2D lox = rowDecimateFilter(commandQueue, image,
                                    level2.h0, {}, &loxEvent);
            cl::Image2D lohi = colDecimateFilter(commandQueue, lox, 
                                    level2.h1, {loxEvent});

            cl::Image2D hilo = rowDecimateFilter(commandQueue, xlo, 
                                    level2.h1, {loEvent});

            cl::Event xbpEvent;
            cl::Image2D xbp = colDecimateFilter(commandQueue, image,
                                    level2.hbp, {}, &xbpEvent);
            cl::Image2D bpbp = rowDecimateFilter(commandQueue, xbp, 
                                    level2.hbp, {xbpEvent});
        }

        lolo = rowDecimateFilter(commandQueue, xlo, level2.h0,
                                 {loEvent}, &loloEvent);

    }


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


