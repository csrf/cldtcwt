#include "dtcwt.h"
#include "clUtil.h"

Dtcwt::Dtcwt(cl::Context& context, const std::vector<cl::Device>& devices)
    : colFilter(context, devices),
      rowFilter(context, devices),
      colDecimateFilter(context, devices),
      rowDecimateFilter(context, devices),
      quadToComplex(context, devices)
{}


// Create the set of images etc needed to perform a DTCWT calculation
DtcwtContext Dtcwt::createContext(size_t imageWidth, size_t imageHeight, 
                                  size_t numLevels, size_t startLevel)
{
    DtcwtContext context;

    // Copy settings to the saved structure
    context.width = imageWidth;
    context.height = imageHeight;

    context.numLevels = numLevels;
    context.startLevel = startLevel;

    // Allocate space on the graphics card
    std::tie(context.outputs,
             context.outputTemps,
             context.noOutputTemps)
        = dummyRun(imageWidth, imageHeight, numLevels, startLevel);

    return context;
}



void Dtcwt::operator() (cl::CommandQueue& commandQueue,
                        cl::Image2D& image, 
                        DtcwtContext& env)
{
    cl::Event xloEvent, loloEvent;

    // Apply the non-decimating, special low pass filters that must be needed
    colFilter(commandQueue, image, env.level1.h0, {},
              &xloEvent, &env.noOutputTemps[0].xlo);
    rowFilter(commandQueue, env.noOutputTemps[0].xlo, env.level1.h0, {xloEvent},
              &loloEvent, &env.noOutputTemps[0].lolo);  

    if (env.startLevel == 0) {
        // Optionally create the parts that are needed, if an output is
        // required
        std::vector<cl::Event> outEvents(
            filter(commandQueue, 
                   image, {},
                   env.noOutputTemps[0].xlo, {xloEvent},
                   &env.outputs[0][0],
                   &env.outputTemps[0],
                   env.level1)
        );
    }


    for (int l = 1; l < env.numLevels; ++l) {

        // The previous low-low has become the new base input
        cl::Event xxEvent = loloEvent;

        // Apply the low pass filters, normal version
        colDecimateFilter(commandQueue, 
                          env.noOutputTemps[l-1].lolo, env.level2.h0, {xxEvent},
                          &xloEvent, &env.noOutputTemps[l].xlo);

        rowDecimateFilter(commandQueue, 
                          env.noOutputTemps[l].xlo, env.level2.h0, {xloEvent},
                          &loloEvent, &env.noOutputTemps[l].lolo);  


        // Produce outputs only when interested in the outcome
        if (l >= env.startLevel) 
            std::vector<cl::Event> outEvents(
                decimateFilter(commandQueue, 
                               env.noOutputTemps[l-1].lolo, {loloEvent},
                               env.noOutputTemps[l].xlo, {xloEvent},
                               &env.outputs[l-env.startLevel][0],
                               &env.outputTemps[l-env.startLevel],
                               env.level2)
            );

    }

}



std::tuple<std::vector<std::vector<cl::Image2D>>,
           std::vector<OutputTemps>,
           std::vector<NoOutputTemps>>
Dtcwt::dummyRun(size_t width, size_t height, int numLevels, int startLevel)
{
    std::vector<std::vector<cl::Image2D>> out;
    std::vector<OutputTemps> outTemps;
    std::vector<NoOutputTemps> noOutTemps;

    noOutTemps.push_back(NoOutputTemps());
    // Apply the non-decimating, special low pass filters that must be needed
    noOutTemps.back().xlo  = colFilter.dummyRun(width, height);
    noOutTemps.back().lolo = rowFilter.dummyRun(noOutTemps.back().xlo);  

    if (startLevel == 0) {
        // Optionally create the parts that are needed, if an output is
        // required
        outTemps.push_back(OutputTemps());
        out.push_back(std::vector<cl::Image2D>(6));
        std::tie(outTemps.back(), out.back())
            = dummyFilter(width, height, noOutTemps.back().xlo);
    }


    for (int l = 1; l < numLevels; ++l) {

        noOutTemps.push_back(NoOutputTemps());
        // Apply the low pass filters, normal version
        noOutTemps.back().xlo  
            = colDecimateFilter.dummyRun((noOutTemps.end()-2)->lolo);
        noOutTemps.back().lolo 
            = rowDecimateFilter.dummyRun(noOutTemps.back().xlo);  

        // Produce outputs only when interested in the outcome
        if (l >= startLevel) {
            outTemps.push_back(OutputTemps());
            out.push_back(std::vector<cl::Image2D>(6));
            std::tie(outTemps.back(), out.back())
                = dummyDecimateFilter((noOutTemps.end()-2)->lolo, 
                                      noOutTemps.back().xlo);
        }
    }

    return std::make_tuple(out, outTemps, noOutTemps);
}



std::tuple<OutputTemps, std::vector<cl::Image2D>>
Dtcwt::dummyFilter(size_t width, size_t height, cl::Image2D xlo)
{
    // Take in the unfiltered and one-way filtered images
    OutputTemps outputTemps;
    std::vector<cl::Image2D> out;

    // Allocate space for the results of filtering
    outputTemps.lox = rowFilter.dummyRun(width, height);
    outputTemps.lohi = colFilter.dummyRun(outputTemps.lox);
    outputTemps.hilo = rowFilter.dummyRun(xlo);
    outputTemps.xbp = colFilter.dummyRun(width, height);
    outputTemps.bpbp = rowFilter.dummyRun(outputTemps.xbp);

    // Space for the subband outputs
    out[2] = quadToComplex.dummyRun(outputTemps.lohi); 
    out[3] = quadToComplex.dummyRun(outputTemps.lohi); 
    out[0] = quadToComplex.dummyRun(outputTemps.lohi); 
    out[5] = quadToComplex.dummyRun(outputTemps.lohi); 
    out[4] = quadToComplex.dummyRun(outputTemps.bpbp);
    out[1] = quadToComplex.dummyRun(outputTemps.bpbp);

    // Temporaries and out subband storage
    return std::tie(outputTemps, out);
}


std::tuple<OutputTemps, std::vector<cl::Image2D>>
Dtcwt::dummyFilter(cl::Image2D xx, cl::Image2D xlo)
{
    // Version for when we are given all images
    return dummyFilter(xx.getImageInfo<CL_IMAGE_WIDTH>(),
                       xx.getImageInfo<CL_IMAGE_HEIGHT>(),
                       xlo);
}



std::vector<cl::Event> Dtcwt::filter(cl::CommandQueue& commandQueue,
                           cl::Image2D& xx, 
                           const std::vector<cl::Event>& xxEvents,
                           cl::Image2D& xlo, 
                           const std::vector<cl::Event>& xloEvents,
                           cl::Image2D* out, 
                           OutputTemps* outputTemps,
                           Filters& filters)
{
    // Low pass one way then high pass the other...
    cl::Event loxEvent, lohiEvent;

    rowFilter(commandQueue, xx, filters.h0, 
              xxEvents, &loxEvent,
              &(outputTemps->lox));

    colFilter(commandQueue, outputTemps->lox, filters.h1, 
              {loxEvent}, &lohiEvent,
              &(outputTemps->lohi));

    // High pass the image that had been low-passed the other way...
    cl::Event hiloEvent;

    rowFilter(commandQueue, xlo, filters.h1, 
                      xloEvents, &hiloEvent,
                      &(outputTemps->hilo));

    // Band pass both ways...
    cl::Event xbpEvent, bpbpEvent;

    colFilter(commandQueue, xx, filters.hbp, 
              xxEvents, &xbpEvent,
              &(outputTemps->xbp));

    rowFilter(commandQueue, outputTemps->xbp, filters.hbp, 
              {xbpEvent}, &bpbpEvent,
              &(outputTemps->bpbp));

    // Create events that, when all done signify everything about this stage
    // is complete
    std::vector<cl::Event> completedEvents(3);

    // ...and generate subband outputs.
    quadToComplex(commandQueue, outputTemps->lohi, 
                  {lohiEvent}, &completedEvents[0], 
                  &out[2], &out[3]);

    quadToComplex(commandQueue, outputTemps->hilo, 
                  {hiloEvent}, &completedEvents[1], 
                  &out[0], &out[5]);

    quadToComplex(commandQueue, outputTemps->bpbp, 
                  {bpbpEvent}, &completedEvents[2], 
                  &out[4], &out[1]);

    return completedEvents;
}



std::tuple<OutputTemps, std::vector<cl::Image2D>>
Dtcwt::dummyDecimateFilter(size_t width, size_t height, cl::Image2D xlo)
{
    // Take in the unfiltered and one-way filtered images

    OutputTemps outputTemps;
    std::vector<cl::Image2D> out;

    // Allocate space for the results of filtering
    outputTemps.lox = rowDecimateFilter.dummyRun(width, height);
    outputTemps.lohi = colDecimateFilter.dummyRun(outputTemps.lox);
    outputTemps.hilo = rowDecimateFilter.dummyRun(xlo);
    outputTemps.xbp = colDecimateFilter.dummyRun(width, height);
    outputTemps.bpbp = rowDecimateFilter.dummyRun(outputTemps.xbp);

    // Space for the subband outputs
    out[2] = quadToComplex.dummyRun(outputTemps.lohi); 
    out[3] = quadToComplex.dummyRun(outputTemps.lohi); 
    out[0] = quadToComplex.dummyRun(outputTemps.lohi); 
    out[5] = quadToComplex.dummyRun(outputTemps.lohi); 
    out[4] = quadToComplex.dummyRun(outputTemps.bpbp);
    out[1] = quadToComplex.dummyRun(outputTemps.bpbp);

    // Temporaries and out subband storage
    return std::tie(outputTemps, out);
}


std::tuple<OutputTemps, std::vector<cl::Image2D>>
Dtcwt::dummyDecimateFilter(cl::Image2D xx, cl::Image2D xlo)
{
    // Version for when we are given all images
    return dummyDecimateFilter(xx.getImageInfo<CL_IMAGE_WIDTH>(),
                               xx.getImageInfo<CL_IMAGE_HEIGHT>(),
                               xlo);
}



std::vector<cl::Event> Dtcwt::decimateFilter(cl::CommandQueue& commandQueue,
                           cl::Image2D& xx, 
                           const std::vector<cl::Event>& xxEvents,
                           cl::Image2D& xlo, 
                           const std::vector<cl::Event>& xloEvents,
                           cl::Image2D* out, 
                           OutputTemps* outputTemps,
                           Filters& filters)
{
    // Low pass one way then high pass the other...
    cl::Event loxEvent, lohiEvent;

    rowDecimateFilter(commandQueue, xx, filters.h0, 
                      xxEvents, &loxEvent,
                      &(outputTemps->lox));

    colDecimateFilter(commandQueue, outputTemps->lox, filters.h1, 
                      {loxEvent}, &lohiEvent,
                      &(outputTemps->lohi));

    // High pass the image that had been low-passed the other way...
    cl::Event hiloEvent;

    rowDecimateFilter(commandQueue, xlo, filters.h1, 
                      xloEvents, &hiloEvent,
                      &(outputTemps->hilo));

    // Band pass both ways...
    cl::Event xbpEvent, bpbpEvent;

    colDecimateFilter(commandQueue, xx, filters.hbp, 
                      xxEvents, &xbpEvent,
                      &(outputTemps->xbp));

    rowDecimateFilter(commandQueue, outputTemps->xbp, filters.hbp, 
                      {xbpEvent}, &bpbpEvent,
                      &(outputTemps->bpbp));

    // Create events that, when all done signify everything about this stage
    // is complete
    std::vector<cl::Event> completedEvents(3);

    // ...and generate subband outputs.
    quadToComplex(commandQueue, outputTemps->lohi, 
                  {lohiEvent}, &completedEvents[0], 
                  &out[2], &out[3]);

    quadToComplex(commandQueue, outputTemps->hilo, 
                  {hiloEvent}, &completedEvents[1], 
                  &out[0], &out[5]);

    quadToComplex(commandQueue, outputTemps->bpbp, 
                  {bpbpEvent}, &completedEvents[2], 
                  &out[4], &out[1]);

    return completedEvents;
}







