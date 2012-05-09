#include "dtcwt.h"
#include "clUtil.h"

Dtcwt::Dtcwt(cl::Context& context, const std::vector<cl::Device>& devices,
             Filters level1, Filters leveln)
    : colFilter(context, devices),
      rowFilter(context, devices),
      colDecimateFilter(context, devices),
      rowDecimateFilter(context, devices),
      quadToComplex(context, devices),
      h0x { context, devices, level1.h0, Filter::x },
      h0y { context, devices, level1.h0, Filter::y },
      h1x { context, devices, level1.h1, Filter::x },
      h1y { context, devices, level1.h1, Filter::y },
      hbpx { context, devices, level1.hbp, Filter::x },
      hbpy { context, devices, level1.hbp, Filter::y },
      g0x { context, devices, leveln.h0, DecimateFilter::x },
      g0y { context, devices, leveln.h0, DecimateFilter::y },
      g1x { context, devices, leveln.h1, DecimateFilter::x },
      g1y { context, devices, leveln.h1, DecimateFilter::y },
      gbpx { context, devices, leveln.hbp, DecimateFilter::x },
      gbpy { context, devices, leveln.hbp, DecimateFilter::y }


{
   
}



// Create the set of images etc needed to perform a DTCWT calculation
DtcwtContext Dtcwt::createContext(size_t imageWidth, size_t imageHeight, 
                                  size_t numLevels, size_t startLevel,
                                  Filters level1In, Filters level2In)
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


    // Take copies of the filters
    context.level1 = level1In;
    context.level2 = level2In;


    return context;
}



void Dtcwt::operator() (cl::CommandQueue& commandQueue,
                        cl::Image2D& image, 
                        DtcwtContext& env)
{
    cl::Event xloEvent, loloEvent;

    // Apply the non-decimating, special low pass filters that must be needed
    h0y(commandQueue, image, env.noOutputTemps[0].xlo, 
        {}, &xloEvent);

    h0x(commandQueue, env.noOutputTemps[0].xlo, env.noOutputTemps[0].lolo,
        {xloEvent}, &loloEvent);


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
        g0x(commandQueue, env.noOutputTemps[l-1].lolo, env.noOutputTemps[l].xlo,
            {xxEvent}, &xloEvent);

        g0y(commandQueue, env.noOutputTemps[l].xlo, env.noOutputTemps[l].lolo,
            {xloEvent}, &loloEvent);  


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



std::tuple<std::vector<Subbands>,
           std::vector<OutputTemps>,
           std::vector<NoOutputTemps>>
Dtcwt::dummyRun(size_t width, size_t height, int numLevels, int startLevel)
{
    std::vector<Subbands> out;
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
        out.push_back(Subbands());
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
            out.push_back(Subbands());
            std::tie(outTemps.back(), out.back())
                = dummyDecimateFilter((noOutTemps.end()-2)->lolo, 
                                      noOutTemps.back().xlo);
        }
    }

    return std::make_tuple(out, outTemps, noOutTemps);
}



std::tuple<OutputTemps, Subbands>
Dtcwt::dummyFilter(size_t width, size_t height, cl::Image2D xlo)
{
    // Take in the unfiltered and one-way filtered images
    OutputTemps outputTemps;
    Subbands out;

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


std::tuple<OutputTemps, Subbands>
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

    h0x(commandQueue, xx, outputTemps->lox,
        xxEvents, &loxEvent);

    h1y(commandQueue, outputTemps->lox, outputTemps->lohi,
        {loxEvent}, &lohiEvent);

    // High pass the image that had been low-passed the other way...
    cl::Event hiloEvent;

    h1x(commandQueue, xlo, outputTemps->hilo,
        xloEvents, &hiloEvent);

    // Band pass both ways...
    cl::Event xbpEvent, bpbpEvent;

    hbpy(commandQueue, xx, outputTemps->xbp,
         xxEvents, &xbpEvent);

    hbpx(commandQueue, outputTemps->xbp, outputTemps->bpbp,
         {xbpEvent}, &bpbpEvent);

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



std::tuple<OutputTemps, Subbands>
Dtcwt::dummyDecimateFilter(size_t width, size_t height, cl::Image2D xlo)
{
    // Take in the unfiltered and one-way filtered images

    OutputTemps outputTemps;
    Subbands out;

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


std::tuple<OutputTemps, Subbands>
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

    g0x(commandQueue, xx, outputTemps->lox,
        xxEvents, &loxEvent);

    g1y(commandQueue, outputTemps->lox, outputTemps->lohi,
        {loxEvent}, &lohiEvent);

    // High pass the image that had been low-passed the other way...
    cl::Event hiloEvent;

    g1x(commandQueue, xlo, outputTemps->hilo,
        xloEvents, &hiloEvent);

    // Band pass both ways...
    cl::Event xbpEvent, bpbpEvent;

    gbpy(commandQueue, xx, outputTemps->xbp,
         xxEvents, &xbpEvent);

    gbpx(commandQueue, outputTemps->xbp, outputTemps->bpbp,
         {xbpEvent}, &bpbpEvent);

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







