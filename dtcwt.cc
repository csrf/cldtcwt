#include "dtcwt.h"
#include "clUtil.h"


static size_t decimateDim(size_t inSize)
{
    // Decimate the size of a dimension by a factor of two.  If this gives
    // a non-even number, pad so it is.
    
    bool pad = (inSize % 4) != 0;

    return inSize / 2 + (pad? 1 : 0);
}



#include <iostream>

SubbandOutputs::SubbandOutputs(const DtcwtEnv& env)
{
    for (int l = env.startLevel; l < env.numLevels; ++l) {

        const cl::Image2D& baseImage = env.levelTemps[l].lolo;

        const size_t width = baseImage.getImageInfo<CL_IMAGE_WIDTH>() / 2,
                    height = baseImage.getImageInfo<CL_IMAGE_HEIGHT>() / 2;

        Subbands sbs;

        std::cout << width << " " << height << std::endl;
        // Create all the complex images at the right size
        for (auto& sb: sbs.sb)
            sb = {env.context_, 0, {CL_RG, CL_FLOAT}, width, height};

        // Add another set of outputs
        subbands.push_back(sbs);
        

    }
}





Dtcwt::Dtcwt(cl::Context& context, const std::vector<cl::Device>& devices,
             Filters level1, Filters leveln)
    : quadToComplex(context, devices),
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
      gbpy { context, devices, leveln.hbp, DecimateFilter::y },
      context_ { context }

{
   
}



// Create the set of images etc needed to perform a DTCWT calculation
DtcwtEnv Dtcwt::createContext(size_t imageWidth, size_t imageHeight, 
                                  size_t numLevels, size_t startLevel)
{
    DtcwtEnv c;

    // Copy settings to the saved structure
    c.context_ = context_;

    c.width = imageWidth;
    c.height = imageHeight;

    c.numLevels = numLevels;
    c.startLevel = startLevel;

    // Make space in advance for the temps
    c.levelTemps.reserve(numLevels);

    // Allocate space on the graphics card for each of the levels
    
    // 1st level
    size_t width = imageWidth + imageWidth % 2;
    size_t height = imageHeight + imageHeight % 2;

    for (int l = 0; l < numLevels; ++l) {

        // Decimate if we're beyond the first stage, otherwise just make
        // sure we have even width/height
        size_t newWidth = 
            (l == 0)? width + width % 2
                    : decimateDim(width);

        size_t newHeight =
            (l == 0)? height + height % 2
                    : decimateDim(height);

        c.levelTemps.push_back(LevelTemps());

        // Temps that will be needed whether there's an output or not
        c.levelTemps.back().xlo
            = createImage2D(context_, width, newHeight);
        c.levelTemps.back().lolo
            = createImage2D(context_, newWidth, newHeight);

        // Temps only needed when producing subband outputs
        if  (l >= startLevel) {
            c.levelTemps.back().lox
                = createImage2D(context_, newWidth, height);
            c.levelTemps.back().lohi
                = createImage2D(context_, newWidth, newHeight);
            c.levelTemps.back().hilo
                = createImage2D(context_, newWidth, newHeight);
            c.levelTemps.back().xbp
                = createImage2D(context_, width, newHeight);
            c.levelTemps.back().bpbp
                = createImage2D(context_, newWidth, newHeight);
        }

        width = newWidth;
        height = newHeight;
    }
   
    return c;
}



void Dtcwt::operator() (cl::CommandQueue& commandQueue,
                        cl::Image2D& image, 
                        DtcwtEnv& env,
                        SubbandOutputs& subbandOutputs)
{
    for (int l = 0; l < env.numLevels; ++l) {

        if (l == 0) {

            filter(commandQueue, image, {},
                   env.levelTemps[l], 
                   (env.startLevel == 0) ? 
                       &subbandOutputs.subbands[0]
                     : nullptr);

        } else {

            decimateFilter(commandQueue, 
                           env.levelTemps[l-1].lolo, 
                               {env.levelTemps[l-1].loloDone},
                           env.levelTemps[l], 
                           (l >= env.startLevel) ? 
                               &subbandOutputs.subbands[l - env.startLevel]
                             : nullptr);

        }

    }

}




void Dtcwt::filter(cl::CommandQueue& commandQueue,
                   cl::Image2D& xx, 
                   const std::vector<cl::Event>& xxEvents,
                   LevelTemps& levelTemps, Subbands* subbands)
{
    // Apply the non-decimating, special low pass filters that must be needed
    h0y(commandQueue, xx, levelTemps.xlo, 
        xxEvents, &levelTemps.xloDone);

    h0x(commandQueue, levelTemps.xlo, levelTemps.lolo,
        {levelTemps.xloDone}, &levelTemps.loloDone);

    // If we've been given subbands to output to, we need to do more work:
    if (subbands) {

        // Low pass one way then high pass the other...
        h0x(commandQueue, xx, levelTemps.lox,
            xxEvents, &levelTemps.loxDone); 
                                            
        h1y(commandQueue, levelTemps.lox, levelTemps.lohi,
            {levelTemps.loxDone}, &levelTemps.lohiDone);

        // High pass the image that had been low-passed the other way...
        h1x(commandQueue, levelTemps.xlo, levelTemps.hilo,
            {levelTemps.xloDone}, &levelTemps.hiloDone);

        // Band pass both ways...
        hbpy(commandQueue, xx, levelTemps.xbp,
             xxEvents, &levelTemps.xbpDone);

        hbpx(commandQueue, levelTemps.xbp, levelTemps.bpbp,
             {levelTemps.xbpDone}, &levelTemps.bpbpDone);

        // Create events that, when all done signify everything about this stage
        // is complete
        subbands->done = std::vector<cl::Event>(3);

        // ...and generate subband outputs.
        quadToComplex(commandQueue, levelTemps.lohi, 
                      {levelTemps.lohiDone}, &subbands->done[0], 
                      &subbands->sb[2], &subbands->sb[3]);

        quadToComplex(commandQueue, levelTemps.hilo, 
                      {levelTemps.hiloDone}, &subbands->done[1], 
                      &subbands->sb[0], &subbands->sb[5]);

        quadToComplex(commandQueue, levelTemps.bpbp, 
                      {levelTemps.bpbpDone}, &subbands->done[2], 
                      &subbands->sb[4], &subbands->sb[1]);
    }
}


void Dtcwt::decimateFilter(cl::CommandQueue& commandQueue,
                           cl::Image2D& xx, 
                           const std::vector<cl::Event>& xxEvents,
                           LevelTemps& levelTemps, Subbands* subbands)
{
    // Apply the non-decimating, special low pass filters that must be needed
    g0y(commandQueue, xx, levelTemps.xlo, 
        xxEvents, &levelTemps.xloDone);

    g0x(commandQueue, levelTemps.xlo, levelTemps.lolo,
        {levelTemps.xloDone}, &levelTemps.loloDone);

    // If we've been given subbands to output to, we need to do more work:
    if (subbands) {

        // Low pass one way then high pass the other...
        g0x(commandQueue, xx, levelTemps.lox,
            xxEvents, &levelTemps.loxDone); 
                                            
        g1y(commandQueue, levelTemps.lox, levelTemps.lohi,
            {levelTemps.loxDone}, &levelTemps.lohiDone);

        // High pass the image that had been low-passed the other way...
        g1x(commandQueue, levelTemps.xlo, levelTemps.hilo,
            {levelTemps.xloDone}, &levelTemps.hiloDone);

        // Band pass both ways...
        gbpy(commandQueue, xx, levelTemps.xbp,
             xxEvents, &levelTemps.xbpDone);

        gbpx(commandQueue, levelTemps.xbp, levelTemps.bpbp,
             {levelTemps.xbpDone}, &levelTemps.bpbpDone);

        // Create events that, when all done signify everything about this stage
        // is complete
        subbands->done = std::vector<cl::Event>(3);

        // ...and generate subband outputs.
        quadToComplex(commandQueue, levelTemps.lohi, 
                      {levelTemps.lohiDone}, &subbands->done[0], 
                      &subbands->sb[2], &subbands->sb[3]);

        quadToComplex(commandQueue, levelTemps.hilo, 
                      {levelTemps.hiloDone}, &subbands->done[1], 
                      &subbands->sb[0], &subbands->sb[5]);

        quadToComplex(commandQueue, levelTemps.bpbp, 
                      {levelTemps.bpbpDone}, &subbands->done[2], 
                      &subbands->sb[4], &subbands->sb[1]);
    }
}





