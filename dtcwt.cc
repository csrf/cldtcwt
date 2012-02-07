#include "dtcwt.h"

Dtcwt::Dtcwt(cl::Context& context, const std::vector<cl::Device>& devices)
    : colFilter(context, devices),
      rowFilter(context, devices),
      colDecimateFilter(context, devices),
      rowDecimateFilter(context, devices),
      quadToComplex(context, devices)
{}


bool operator== (const DtcwtParams& params1,
                  const DtcwtParams& params2)
{
    return (params1.width == params2.width)
        && (params1.height == params2.height)
        && (params1.numLevels == params2.numLevels)
        && (params1.startLevel == params2.startLevel);
}





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
    }

    cl::Image2D lolo = rowFilter(commandQueue, xlo, level1.h0, 
                                 {loEvent}, &loloEvent);

    for (int l = 1; l < numLevels; ++l) {

        xlo = colDecimateFilter(commandQueue, lolo, level2.h0,
                                {loloEvent}, &loEvent);

        // High pass only when interested in the outcome
        if (l >= startLevel) {
                    }

        lolo = rowDecimateFilter(commandQueue, xlo, level2.h0,
                                 {loEvent}, &loloEvent);

    }

    return result;
}



std::tuple<std::vector<std::vector<cl::Image2D>>,
           std::vector<OutputTemps>,
           std::vector<NoOutputTemps>>
Dtcwt::dummyRun(cl::Image2D image, int numLevels, int startLevel)
{
    std::vector<std::vector<cl::Image2D>> out;
    std::vector<OutputTemps> outTemps;
    std::vector<NoOutputTemps> noOutTemps;

    noOutTemps.push_back(NoOutputTemps());
    noOutTemps[0].xlo = colFilter.dummyRun(image);
    noOutTemps[0].lolo = rowFilter.dummyRun(noOutTemps[0].xlo);  

    if (startLevel == 0) {
        outTemps.push_back(OutputTemps());
        out.push_back(std::vector<cl::Image2D>(6));
        std::tie(outTemps.back(), out.back())
            = dummyFilter(image, noOutTemps[0].xlo);
    }


    for (int l = 1; l < numLevels; ++l) {

        noOutTemps.push_back(NoOutputTemps());
        noOutTemps[l].xlo  = colFilter.dummyRun(noOutTemps[l-1].lolo);
        noOutTemps[l].lolo = rowFilter.dummyRun(noOutTemps[l].xlo);  

        // High pass only when interested in the outcome
        if (l >= startLevel) {
            outTemps.push_back(OutputTemps());
            out.push_back(std::vector<cl::Image2D>(6));
            std::tie(outTemps.back(), out.back())
                = dummyFilter(image, noOutTemps[0].xlo);
        }
    }

    return std::make_tuple(out, outTemps, noOutTemps);
}


std::tuple<OutputTemps, std::vector<cl::Image2D>>
Dtcwt::dummyDecimateFilter(cl::Image2D xx, cl::Image2D xlo)
{
    OutputTemps outputTemps;
    std::vector<cl::Image2D> out;

    outputTemps.lox = rowDecimateFilter.dummyRun(xx);
    outputTemps.lohi = colDecimateFilter.dummyRun(outputTemps.lox);

    out[2] = quadToComplex.dummyRun(outputTemps.lohi); 
    out[3] = quadToComplex.dummyRun(outputTemps.lohi); 

    outputTemps.hilo = rowDecimateFilter.dummyRun(xlo);
    out[0] = quadToComplex.dummyRun(outputTemps.lohi); 
    out[5] = quadToComplex.dummyRun(outputTemps.lohi); 

    outputTemps.xbp = colDecimateFilter.dummyRun(xx);
    outputTemps.bpbp = rowDecimateFilter.dummyRun(outputTemps.xbp);
    out[4] = quadToComplex.dummyRun(outputTemps.bpbp);
    out[1] = quadToComplex.dummyRun(outputTemps.bpbp);

    return std::tie(outputTemps, out);
}


std::tuple<OutputTemps, std::vector<cl::Image2D>>
Dtcwt::dummyFilter(cl::Image2D xx, cl::Image2D xlo)
{
    OutputTemps outputTemps;
    std::vector<cl::Image2D> out;

    outputTemps.lox = rowFilter.dummyRun(xx);
    outputTemps.lohi = colFilter.dummyRun(outputTemps.lox);
    out[2] = quadToComplex.dummyRun(outputTemps.lohi); 
    out[3] = quadToComplex.dummyRun(outputTemps.lohi); 

    outputTemps.hilo = rowFilter.dummyRun(xlo);
    out[0] = quadToComplex.dummyRun(outputTemps.lohi); 
    out[5] = quadToComplex.dummyRun(outputTemps.lohi); 

    outputTemps.xbp = colFilter.dummyRun(xx);
    outputTemps.bpbp = rowFilter.dummyRun(outputTemps.xbp);
    out[4] = quadToComplex.dummyRun(outputTemps.bpbp);
    out[1] = quadToComplex.dummyRun(outputTemps.bpbp);

    return std::tie(outputTemps, out);
}


void Dtcwt::decimateFilter(cl::Image2D& xx, cl::Event xxEvent,
                           cl::Image2D& xlo, cl::Event xloEvent,
                           cl::Image2D* out, 
                           OutputTemps* outputTemps)
{
    cl::Event loxEvent, lohiEvent;
    rowDecimateFilter(commandQueue, xx, level2.h0, 
                      {}, &loxEvent,
                      &(outputTemps->lox));
    colDecimateFilter(commandQueue, outputTemps->lox, level2.h1, 
                      {loxEvent}, &lohiEvent,
                      &(outputTemps->lohi));

    quadToComplex(commandQueue, outputTemps->lohi, 
                  {lohiEvent}, 0, 
                  &out[2], &out[3]);

    cl::Event hiloEvent;
    rowDecimateFilter(commandQueue, xlo, level2.h1, 
                      {loEvent}, &hiloEvent,
                      &(outputTemps->hilo));

    quadToComplex(commandQueue, hilo, {hiloEvent}, 0, &out[0], &out[5]);

    cl::Event xbpEvent, bpbpEvent;
    colDecimateFilter(commandQueue, image, level2.hbp, 
                      {}, &xbpEvent,
                      &(outputTemps->xbp));
    rowDecimateFilter(commandQueue, xbp, level2.hbp, 
                      {xbpEvent}, &bpbpEvent,
                      &(outputTemps->bpbp));

    quadToComplex(commandQueue, bpbp, {bpbpEvent}, 0, &out[4], &out[1]);
}


cl::Image2D Dtcwt::filter(cl::Image2D& xx, cl::Image2D xlo,
                                      )
{
    cl::Event loxEvent, lohiEvent;
    cl::Image2D lox = rowFilter(commandQueue, image, level1.h0,
                                {}, &loxEvent);
    cl::Image2D lohi = colFilter(commandQueue, lox, level1.h1,
                                 {loxEvent} &lohiEvent);

    std::tie(out[2], out[3]) = quadToComplex(commandQueue, lohi, 
                                           {lohiEvent});

    cl::Event hiloEvent;
    cl::Image2D hilo = rowFilter(commandQueue, xlo, level1.h1,
                                 {loEvent}, &hiloEvent);

    std::tie(out[0], out[5]) = quadToComplex(commandQueue, hilo, 
                                           {hiloEvent});

    cl::Event xbpEvent, bpbpEvent;
    cl::Image2D xbp = colFilter(commandQueue, image, level1.hbp,
                                {}, &xbpEvent);
    cl::Image2D bpbp = rowFilter(commandQueue, xbp, level1.hbp,
                                 {xbpEvent}, &bpbpEvent);

    std::tie(out[4], out[1]) = quadToComplex(commandQueue, bpbp, 
                                           {bpbpEvent});

}


