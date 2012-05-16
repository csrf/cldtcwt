#include "dtcwt.h"
#include "clUtil.h"


static size_t decimateDim(size_t inSize)
{
    // Decimate the size of a dimension by a factor of two.  If this gives
    // a non-even number, pad so it is.
    
    bool pad = (inSize % 4) != 0;

    return inSize / 2 + (pad? 1 : 0);
}




DtcwtOutput::DtcwtOutput(const DtcwtTemps& env)
{
    for (int l = env.startLevel; l < env.numLevels; ++l) {

        const cl::Image2D& baseImage = env.levelTemps[l].lolo;

        const size_t width = baseImage.getImageInfo<CL_IMAGE_WIDTH>() / 2,
                    height = baseImage.getImageInfo<CL_IMAGE_HEIGHT>() / 2;

        LevelOutput sbs;

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
DtcwtTemps Dtcwt::createContext(size_t imageWidth, size_t imageHeight, 
                                  size_t numLevels, size_t startLevel)
{
    DtcwtTemps c;

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
                        DtcwtTemps& env,
                        DtcwtOutput& subbandOutputs)
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
                   LevelTemps& levelTemps, LevelOutput* subbands)
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
                      subbands->sb[2], subbands->sb[3],
                      {levelTemps.lohiDone}, &subbands->done[0]);

        quadToComplex(commandQueue, levelTemps.hilo, 
                      subbands->sb[0], subbands->sb[5],
                      {levelTemps.hiloDone}, &subbands->done[1]);

        quadToComplex(commandQueue, levelTemps.bpbp, 
                      subbands->sb[4], subbands->sb[1],
                      {levelTemps.bpbpDone}, &subbands->done[2]);
    }
}


void Dtcwt::decimateFilter(cl::CommandQueue& commandQueue,
                           cl::Image2D& xx, 
                           const std::vector<cl::Event>& xxEvents,
                           LevelTemps& levelTemps, LevelOutput* subbands)
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
                      subbands->sb[2], subbands->sb[3],
                      {levelTemps.lohiDone}, &subbands->done[0]); 

        quadToComplex(commandQueue, levelTemps.hilo, 
                      subbands->sb[0], subbands->sb[5],
                      {levelTemps.hiloDone}, &subbands->done[1]); 

        quadToComplex(commandQueue, levelTemps.bpbp, 
                      subbands->sb[4], subbands->sb[1],
                      {levelTemps.bpbpDone}, &subbands->done[2]); 
    }
}



#include <iostream>



EnergyMap::EnergyMap(cl::Context& context,
                     const std::vector<cl::Device>& devices)
   : context_(context)
{
    // The OpenCL kernel:
    const std::string sourceCode = 
        "__kernel void energyMap(__read_only image2d_t sb0,"
                                "__read_only image2d_t sb1,"
                                "__read_only image2d_t sb2,"
                                "__read_only image2d_t sb3,"
                                "__read_only image2d_t sb4,"
                                "__read_only image2d_t sb5,"
                                "__write_only image2d_t out)"
        "{"
            "sampler_t s ="
                "CLK_NORMALIZED_COORDS_FALSE"
                "| CLK_ADDRESS_CLAMP"
                "| CLK_FILTER_NEAREST;"

            "int x = get_global_id(0);"
            "int y = get_global_id(1);"

            "if (x < get_image_width(out)"
             "&& y < get_image_height(out)) {"

                // Sample each subband
                "float2 h0 = read_imagef(sb0, s, (int2) (x,y)).s01;"
                "float2 h1 = read_imagef(sb1, s, (int2) (x,y)).s01;"
                "float2 h2 = read_imagef(sb2, s, (int2) (x,y)).s01;"
                "float2 h3 = read_imagef(sb3, s, (int2) (x,y)).s01;"
                "float2 h4 = read_imagef(sb4, s, (int2) (x,y)).s01;"
                "float2 h5 = read_imagef(sb5, s, (int2) (x,y)).s01;"

                // Convert to absolute (still squared, because it's more
                // convenient)
                "float abs_h0_2 = h0.s0 * h0.s0 + h0.s1 * h0.s1;"
                "float abs_h1_2 = h1.s0 * h1.s0 + h1.s1 * h1.s1;"
                "float abs_h2_2 = h2.s0 * h2.s0 + h2.s1 * h2.s1;"
                "float abs_h3_2 = h3.s0 * h3.s0 + h3.s1 * h3.s1;"
                "float abs_h4_2 = h4.s0 * h4.s0 + h4.s1 * h4.s1;"
                "float abs_h5_2 = h5.s0 * h5.s0 + h5.s1 * h5.s1;"

                // Calculate result
                "float result ="
                    "(  sqrt(abs_h0_2 * abs_h3_2) "
                    " + sqrt(abs_h1_2 * abs_h4_2) "
                    " + sqrt(abs_h2_2 * abs_h5_2))"
                    "/"
                    "sqrt(0.001 + "
                    "   1.5 * (  abs_h0_2 + abs_h1_2 + abs_h2_2"
                               " + abs_h3_2 + abs_h4_2 + abs_h5_2));" 

                // Produce output
                "write_imagef(out, (int2) (x, y), result);"

            "}"

        "}";

    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(std::make_pair(sourceCode.c_str(), sourceCode.length()));

    // Compile it...
    cl::Program program(context, source);

    try {
        program.build(devices);
    } catch(cl::Error err) {
	    std::cerr 
		    << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
		    << std::endl;
	    throw;
    } 
        
    // ...and extract the useful part, viz the kernel
    kernel_ = cl::Kernel(program, "energyMap");

}


static int roundWGs(int l, int lWG)
{
    return lWG * (l / lWG + ((l % lWG) ? 1 : 0)); 
}


void EnergyMap::operator() (cl::CommandQueue& commandQueue,
                            const LevelOutput& levelOutput,
                            cl::Image2D& energyMap,
                            cl::Event* doneEvent)
{
    // Set up all the arguments to the kernel
    for (int n = 0; n < levelOutput.sb.size(); ++n)
        kernel_.setArg(n, levelOutput.sb[n]);

    kernel_.setArg(levelOutput.sb.size(), energyMap);

    const size_t wgSize = 16;

    cl::NDRange globalSize = {
        roundWGs(energyMap.getImageInfo<CL_IMAGE_WIDTH>(), wgSize),
        roundWGs(energyMap.getImageInfo<CL_IMAGE_HEIGHT>(), wgSize)
    };

    // Execute
    commandQueue.enqueueNDRangeKernel(kernel_, cl::NullRange,
                                      globalSize,
                                      {wgSize, wgSize},
                                      &levelOutput.done, doneEvent);
}


















