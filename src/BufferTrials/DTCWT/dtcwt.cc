#include "dtcwt.h"
#include <cmath>

#include "util/clUtil.h"


// First level coefficients
std::vector<float> h0oCoefs(float scaleFactor);
std::vector<float> h1oCoefs(float scaleFactor);
std::vector<float> h2oCoefs(float scaleFactor);

// Decimation coefficinets
std::vector<float> h0bCoefs(float scaleFactor);
std::vector<float> h1bCoefs(float scaleFactor);
std::vector<float> h2bCoefs(float scaleFactor);


static size_t decimateDim(size_t inSize)
{
    // Decimate the size of a dimension by a factor of two.  If this gives
    // a non-even number, pad so it is.
    
    bool pad = (inSize % 4) != 0;

    return inSize / 2 + (pad? 1 : 0);
}





Dtcwt::Dtcwt(cl::Context& context, const std::vector<cl::Device>& devices,
             float scaleFactor) : 

    context_ {context},

    // Non-decimating
    h0ox {context, devices, h0oCoefs(scaleFactor)},
    h1ox {context, devices, h1oCoefs(scaleFactor)},
    h2ox {context, devices, h2oCoefs(scaleFactor)},

    h0oy {context, devices, h0oCoefs(scaleFactor)},
    h1oy {context, devices, h1oCoefs(scaleFactor)},
    h2oy {context, devices, h2oCoefs(scaleFactor)},

    quadToComplex {context, devices},

    // Decimating
    h0bx {context, devices, h0bCoefs(scaleFactor), false},
    h0by {context, devices, h0bCoefs(scaleFactor), false},

    // 3-way decimating filter
    h012bx {context, devices, h0bCoefs(scaleFactor), false,
                              h1bCoefs(scaleFactor), true,
                              h2bCoefs(scaleFactor), true},

    // Filtering, decimation, complex conversion
    q2ch0by {context, devices, h0bCoefs(scaleFactor), false},
    q2ch1by {context, devices, h1bCoefs(scaleFactor), true},
    q2ch2by {context, devices, h2bCoefs(scaleFactor), true}
{}




DtcwtOutput::DtcwtOutput(const DtcwtTemps& env)
{
    for (int l = env.startLevel; l < env.numLevels; ++l) {

        const ImageBuffer& baseImage = env.levelTemps[l].lolo;

        const size_t width = baseImage.width() / 2,
                    height = baseImage.height() / 2;

        LevelOutput sbs;

        // Create all the complex images at the right size
        for (auto& sb: sbs.sb)
            sb = {env.context_, 0, {CL_RG, CL_FLOAT}, width, height};

        // Add another set of outputs
        subbands.push_back(sbs);
        

    }
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
        c.levelTemps.back().lo = ImageBuffer(context_, CL_MEM_READ_WRITE,
                                             newWidth, height, 
                                             padding_, alignment_);

        c.levelTemps.back().lolo = ImageBuffer(context_, CL_MEM_READ_WRITE,
                                               newWidth, newHeight, 
                                               padding_, alignment_);
       
        // Temps only needed when producing subband outputs
        if  (l >= startLevel) {

            c.levelTemps.back().hi
                = ImageBuffer(context_, CL_MEM_READ_WRITE,
                              newWidth, height, 
                              padding_, alignment_);

            c.levelTemps.back().bp
                = ImageBuffer(context_, CL_MEM_READ_WRITE,
                              newWidth, height, 
                              padding_, alignment_);

            // We need more intermediate images if producing outputs
            // at Level 1
            if (l == 0) {

                c.levelTemps.back().lohi
                    = ImageBuffer(context_, CL_MEM_READ_WRITE,
                                  newWidth, newHeight, 
                                  padding_, alignment_);

                c.levelTemps.back().hilo
                    = ImageBuffer(context_, CL_MEM_READ_WRITE,
                                  newWidth, newHeight, 
                                  padding_, alignment_);

                c.levelTemps.back().bpbp
                    = ImageBuffer(context_, CL_MEM_READ_WRITE,
                                  newWidth, newHeight, 
                                  padding_, alignment_);

            }
        }

        width = newWidth;
        height = newHeight;
    }
   
    return c;
}



void Dtcwt::operator() (cl::CommandQueue& commandQueue,
                        ImageBuffer& image, 
                        DtcwtTemps& env,
                        DtcwtOutput& subbandOutputs,
                        const std::vector<cl::Event>& waitEvents)
{
    for (int l = 0; l < env.numLevels; ++l) {

        if (l == 0) {

            filter(commandQueue, image, waitEvents,
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
                   ImageBuffer& xx, 
                   const std::vector<cl::Event>& xxEvents,
                   LevelTemps& levelTemps, LevelOutput* subbands)
{
    // Apply the non-decimating, special low pass filters that must be needed
    h0ox(commandQueue, xx, levelTemps.lo, 
         xxEvents, &levelTemps.loDone);

    h0oy(commandQueue, levelTemps.lo, levelTemps.lolo,
         {levelTemps.loDone}, &levelTemps.loloDone);

    // If we've been given subbands to output to, we need to do more work:
    if (subbands) {

        // Produce both the other vertically-filtered versions
        h1ox(commandQueue, xx, levelTemps.hi,
             xxEvents, &levelTemps.hiDone);

        h2ox(commandQueue, xx, levelTemps.bp,
             xxEvents, &levelTemps.bpDone);

        // High pass the images that had been low-passed the other way
        h0oy(commandQueue, levelTemps.hi, levelTemps.lohi,
             {levelTemps.hiDone}, &levelTemps.lohiDone);

        h1oy(commandQueue, levelTemps.lo, levelTemps.hilo,
             {levelTemps.loDone}, &levelTemps.hiloDone);

        h2oy(commandQueue, levelTemps.bp, levelTemps.bpbp,
             {levelTemps.bpDone}, &levelTemps.bpbpDone);

        // Create events that, when all done signify everything about this stage
        // is complete
        subbands->done = std::vector<cl::Event>(3);

        // ...and generate subband outputs.
        quadToComplex(commandQueue, levelTemps.lohi, 
                      subbands->sb[0], subbands->sb[5],
                      {levelTemps.lohiDone}, &subbands->done[0]); 

        quadToComplex(commandQueue, levelTemps.hilo, 
                      subbands->sb[2], subbands->sb[3],
                      {levelTemps.hiloDone}, &subbands->done[1]); 

        quadToComplex(commandQueue, levelTemps.bpbp, 
                      subbands->sb[1], subbands->sb[4],
                      {levelTemps.bpbpDone}, &subbands->done[2]); 
 
    }
}


void Dtcwt::decimateFilter(cl::CommandQueue& commandQueue,
                           ImageBuffer& xx, 
                           const std::vector<cl::Event>& xxEvents,
                           LevelTemps& levelTemps, LevelOutput* subbands)
{
    if (subbands == nullptr) {

        // Apply the non-decimating, low-pass filters both ways
        h0bx(commandQueue, xx, levelTemps.lo, 
             xxEvents, &levelTemps.loDone);

        h0by(commandQueue, levelTemps.lo, levelTemps.lolo,
             {levelTemps.loDone}, &levelTemps.loloDone);


    } else {
        // If we've been given subbands to output to, we need to do more work:

        // Produce all the vertically-filtered versions
        h012bx(commandQueue, xx, 
               levelTemps.lo,
               levelTemps.hi,
               levelTemps.bp,
               xxEvents, &levelTemps.loDone);

        // Prepare low-low output
        h0by(commandQueue, levelTemps.lo, levelTemps.lolo,
             {levelTemps.loDone}, &levelTemps.loloDone);

        // Create events that, when all done signify everything about this stage
        // is complete
        subbands->done = std::vector<cl::Event>(3);

        // ...and filter in the y direction, generating subband outputs.
        q2ch1by(commandQueue, levelTemps.lo, 
                subbands->sb[0], subbands->sb[5],
                {levelTemps.loDone}, &subbands->done[0]); 

        q2ch0by(commandQueue, levelTemps.hi, 
                subbands->sb[2], subbands->sb[3],
                {levelTemps.loDone}, &subbands->done[1]); 

        q2ch2by(commandQueue, levelTemps.bp, 
                subbands->sb[1], subbands->sb[4],
                {levelTemps.loDone}, &subbands->done[2]); 
    }
}






// First level coefficients
std::vector<float> h0oCoefs(float scaleFactor)
{
    std::vector<float> h = { 
          -0.001757812500000,
           0.000000000000000,
           0.022265625000000,
          -0.046875000000000,
          -0.048242187500000,
           0.296875000000000,
           0.555468750000000,
           0.296875000000000,
          -0.048242187500000,
          -0.046875000000000,
           0.022265625000000,
           0.000000000000000,
          -0.001757812500000
    };

    // Scale so that when applied in both directions gives the correct
    // overall scale factor
    for (float& val: h)
        val *= sqrt(scaleFactor);

    return h;
}



std::vector<float> h1oCoefs(float scaleFactor)
{
    std::vector<float> h = { 
//          -0.000070626395089,
 //          0.000000000000000,
           0.001341901506696,
          -0.001883370535714,
          -0.007156808035714,
           0.023856026785714,
           0.055643136160714,
          -0.051688058035714,
          -0.299757603236607,
           0.559430803571429,
          -0.299757603236607,
          -0.051688058035714,
           0.055643136160714,
           0.023856026785714,
          -0.007156808035714,
          -0.001883370535714,
           0.001341901506696,
  //         0.000000000000000,
   //       -0.000070626395089
   };

    // Scale so that when applied in both directions gives the correct
    // overall scale factor
    for (float& val: h)
        val *= sqrt(scaleFactor);

    return h;
}



std::vector<float> h2oCoefs(float scaleFactor)
{
    std::vector<float> h = { 
 //         -3.68250025673202e-04,
  //        -6.22253585579744e-04,
          -7.81782479825950e-05,
           4.18582084706810e-03,
           8.19178717888364e-03,
          -7.42327402480263e-03,
          -6.15384268799117e-02,
          -1.48158230911691e-01,
          -1.17076301639216e-01,
           6.52908215843590e-01,
          -1.17076301639216e-01,
          -1.48158230911691e-01,
          -6.15384268799117e-02,
          -7.42327402480263e-03,
           8.19178717888364e-03,
           4.18582084706810e-03,
          -7.81782479825949e-05,
   //       -6.22253585579744e-04,
    //      -3.68250025673202e-04
    };


    // Scale so that when applied in both directions gives the correct
    // overall scale factor
    for (float& val: h)
        val *= sqrt(scaleFactor);

    return h;
}



// Decimation coefficinets
std::vector<float> h0bCoefs(float scaleFactor)
{
    std::vector<float> h = { 
          -0.00455689562847549,
          -0.00543947593727412,
           0.01702522388155399,
           0.02382538479492030,
          -0.10671180468666540,
           0.01186609203379700,
           0.56881042071212273,
           0.75614564389252248,
           0.27529538466888204,
          -0.11720388769911527,
          -0.03887280126882779,
           0.03466034684485349,
          -0.00388321199915849,
           0.00325314276365318
    };


    // Scale so that when applied in both directions gives the correct
    // overall scale factor
    for (float& val: h)
        val *= sqrt(scaleFactor);

    return h;
}


std::vector<float> h1bCoefs(float scaleFactor)
{
    std::vector<float> h = { 
          -0.00325314276365318,
          -0.00388321199915849,
          -0.03466034684485349,
          -0.03887280126882779,
           0.11720388769911527,
           0.27529538466888204,
          -0.75614564389252248,
           0.56881042071212273,
          -0.01186609203379700,
          -0.10671180468666540,
          -0.02382538479492030,
           0.01702522388155399,
           0.00543947593727412,
          -0.00455689562847549
    };


    // Scale so that when applied in both directions gives the correct
    // overall scale factor
    for (float& val: h)
        val *= sqrt(scaleFactor);

    return h;
}


std::vector<float> h2bCoefs(float scaleFactor)
{
    std::vector<float> h = { 
          -2.77165349347537e-03,
          -4.32919303381105e-04,
           2.10100577283097e-02,
           6.14446533755929e-02,
           1.73241472867428e-01,
          -4.47647940175083e-02,
          -8.38137840090472e-01,
           4.36787385780317e-01,
           2.62691880616686e-01,
          -7.62474758151248e-03,
          -2.63685613793659e-02,
          -2.54554351814246e-02,
          -9.59514305416110e-03,
          -2.43562670333119e-05
    };


    // Scale so that when applied in both directions gives the correct
    // overall scale factor
    for (float& val: h)
        val *= sqrt(scaleFactor);

    return h;
}







