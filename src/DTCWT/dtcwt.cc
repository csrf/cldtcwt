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






// LevelTemps functions

LevelTemps::LevelTemps()
    : inputWidth_(0), inputHeight_(0), 
      outputWidth_(0), outputHeight_(0), 
      isLevelOne_(false),
      producesOutputs_(false)
{
    // Default constructor, so that an uninitialised DtcwtTemps
    // will not cause problems if declared on its own.
}



LevelTemps::LevelTemps(cl::Context& context,
                       size_t inputWidth, size_t inputHeight,
                       size_t padding, size_t alignment,
                       bool isLevelOne,
                       bool producesOutputs)
 : inputWidth_(inputWidth), inputHeight_(inputHeight), 
   isLevelOne_(isLevelOne), producesOutputs_(producesOutputs)
{
    // Dimensions provided are for the input
    
    // If we're at level one, we do not decimate
    // This way we also deal with odd-sized images
    outputWidth_  = isLevelOne_? (inputWidth_ +  (inputWidth_ & 1))
                               : decimateDim(inputWidth_);
    outputHeight_ = isLevelOne_? (inputHeight_ + (inputHeight_ & 1))
                               : decimateDim(inputHeight_);

    // x-filtered version
    lo = ImageBuffer<cl_float>
                    (context, CL_MEM_READ_WRITE,
                     outputWidth_, inputHeight_, 
                     padding, alignment,
                     producesOutputs_? 3 : 1);

    // x & y filtered version
    lolo = ImageBuffer<cl_float>
                      (context, CL_MEM_READ_WRITE,
                       outputWidth_, outputHeight_, 
                       padding, alignment);
 
    if (producesOutputs_) {

        // These are the versions that have been filtered in the
        // x-direction (along rows), ready to be filtered along y and
        // produce outputs.
        bp = ImageBuffer<cl_float>(lo, 1);
        hi = ImageBuffer<cl_float>(lo, 2);

        if (isLevelOne_) {
            // Level one, when producing outputs, needs some extra
            // intermediates because complex conversion isn't rolled into
            // the filter
            lohi = ImageBuffer<cl_float>
                         (context, CL_MEM_READ_WRITE,
                          outputWidth_, outputHeight_, 
                          padding, alignment);

            hilo = ImageBuffer<cl_float>
                         (context, CL_MEM_READ_WRITE,
                          outputWidth_, outputHeight_, 
                          padding, alignment);

            bpbp = ImageBuffer<cl_float>
                         (context, CL_MEM_READ_WRITE,
                          outputWidth_, outputHeight_, 
                          padding, alignment);
        }

    }
}






// Create the set of images etc needed to perform a DTCWT calculation
DtcwtTemps::DtcwtTemps(cl::Context& context,
                       size_t imageWidth, size_t imageHeight, 
                       size_t startLevel, size_t numLevels)
  : context_(context),
    width_(imageWidth), height_(imageHeight),
    startLevel_(startLevel), numLevels_(numLevels)
{
    // Make space in advance for the temps
    levelTemps_.reserve(numLevels);

    // Allocate space on the graphics card for each of the levels
    
    // 1st level
    size_t width  = imageWidth;
    size_t height = imageHeight;

    for (int l = 1; l < (startLevel + numLevels); ++l) {

        levelTemps_.emplace_back(context_, width, height,
                                 padding_, alignment_,
                                 l == 1, l >= startLevel);

        width  = levelTemps_.back().outputWidth_;
        height = levelTemps_.back().outputHeight_;
    }
}



DtcwtOutput DtcwtTemps::createOutputs()
{
    // Construct an output structure, using the sizes we already know

    DtcwtOutput output;

    output.startLevel_ = startLevel_;
    output.numLevels_ = numLevels_;

    for (const auto& levelTemp: levelTemps_)
        if (levelTemp.producesOutputs_) {

            output.levels_.emplace_back(context_, 
                    CL_MEM_READ_WRITE,
                    levelTemp.outputWidth_ / 2,
                    levelTemp.outputHeight_ / 2,
                    0, 1,
                    6);

            // Add a three-long vector to the list of wait events
            output.doneEvents_.emplace_back(3);

        }

    return output;

}




Subbands& DtcwtOutput::level(int levelNum)
{
    return levels_[levelNum-startLevel_];
}


const Subbands& DtcwtOutput::level(int levelNum) const
{
    return levels_[levelNum-startLevel_];
}


std::vector<cl::Event> DtcwtOutput::doneEvents(int levelNum)
{
    return doneEvents_[levelNum - startLevel_];
}


const std::vector<cl::Event> DtcwtOutput::doneEvents(int levelNum) const
{
    return doneEvents_[levelNum - startLevel_];
}



Subbands& DtcwtOutput::operator [] (int n)
{
    return levels_[n];
}


const Subbands& DtcwtOutput::operator [] (int n) const
{
    return levels_[n];
}


std::vector<Subbands>::iterator DtcwtOutput::begin()
{
    return levels_.begin();
}


std::vector<Subbands>::const_iterator DtcwtOutput::begin() const
{
    return levels_.begin();
}


std::vector<Subbands>::iterator DtcwtOutput::end()
{
    return levels_.end();
}


std::vector<Subbands>::const_iterator DtcwtOutput::end() const
{
    return levels_.end();
}



size_t DtcwtOutput::startLevel() const
{
    return startLevel_;
}


size_t DtcwtOutput::numLevels() const
{
    return numLevels_;
}




Dtcwt::Dtcwt(cl::Context& context, const std::vector<cl::Device>& devices,
             float scaleFactor) : 

    context_ {context},

    // Pad by symmetric extension
    padX {context, devices},
    padY {context, devices}, 

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
    h021bx {context, devices, h0bCoefs(scaleFactor), false,
                              h2bCoefs(scaleFactor), true,
                              h1bCoefs(scaleFactor), true},

    // Filtering, decimation, complex conversion
    q2c_h1_h2_h0 {context, devices,
                  h1bCoefs(scaleFactor), true,
                  h2bCoefs(scaleFactor), true,
                  h0bCoefs(scaleFactor), false}
{}






void Dtcwt::operator() (cl::CommandQueue& commandQueue,
                        ImageBuffer<cl_float>& image, 
                        DtcwtTemps& temps,
                        DtcwtOutput& output,
                        const std::vector<cl::Event>& waitEvents)
{
    int outputIdx = 0;

    for (int l = 0; l < temps.levelTemps_.size(); ++l) {

        if (l == 0) {

            filter(commandQueue, image, waitEvents,
                   temps.levelTemps_[l], 
                   temps.levelTemps_[l].producesOutputs_? 
                       &output.levels_[0]
                     : nullptr,
                   temps.levelTemps_[l].producesOutputs_? 
                       &output.doneEvents_[0]
                     : nullptr);

        } else {

            decimateFilter(commandQueue, 
                           temps.levelTemps_[l-1].lolo, 
                               {temps.levelTemps_[l-1].loloDone},
                           temps.levelTemps_[l], 
                           temps.levelTemps_[l].producesOutputs_? 
                               &output.levels_[outputIdx] : nullptr,
                           temps.levelTemps_[l].producesOutputs_? 
                               &output.doneEvents_[outputIdx] : nullptr);

        }
        
        if (temps.levelTemps_[l].producesOutputs_)
            ++outputIdx;

    }

}




void Dtcwt::filter(cl::CommandQueue& commandQueue,
                   ImageBuffer<cl_float>& xx, 
                   const std::vector<cl::Event>& xxEvents,
                   LevelTemps& levelTemps, 
                   Subbands* subbands,
                   std::vector<cl::Event>* events)
{
    // Events are the events which, when done, signal that the Subband
    // outputs are complete

    // Definitely need to do this padding, whether outputs or not
    cl::Event xxPadded;
    padX(commandQueue, xx, xxEvents, &xxPadded);

    // Apply the non-decimating, special low pass filters that must be needed
    h0ox(commandQueue, xx, levelTemps.lo, 
         {xxPadded}, &levelTemps.loDone);

    cl::Event loPadded;
    padY(commandQueue, levelTemps.lo, {levelTemps.loDone}, &loPadded);

    h0oy(commandQueue, levelTemps.lo, levelTemps.lolo,
         {loPadded}, &levelTemps.loloDone);

    // If we've been given subbands to output to, we need to do more work:
    if (subbands) {

        // Produce both the other vertically-filtered versions
        h1ox(commandQueue, xx, levelTemps.hi,
             {xxPadded}, &levelTemps.hiDone);

        h2ox(commandQueue, xx, levelTemps.bp,
             {xxPadded}, &levelTemps.bpDone);

        cl::Event hiPadded, bpPadded;
        padY(commandQueue, levelTemps.hi, {levelTemps.hiDone}, &hiPadded);
        padY(commandQueue, levelTemps.bp, {levelTemps.bpDone}, &bpPadded);

        // High pass the images that had been low-passed the other way
        h0oy(commandQueue, levelTemps.hi, levelTemps.lohi,
             {hiPadded}, &levelTemps.lohiDone);

        h1oy(commandQueue, levelTemps.lo, levelTemps.hilo,
             {loPadded}, &levelTemps.hiloDone);

        h2oy(commandQueue, levelTemps.bp, levelTemps.bpbp,
             {bpPadded}, &levelTemps.bpbpDone);

        // Create events that, when all done signify everything about this stage
        // is complete
        *events = std::vector<cl::Event>(3);

        // ...and generate subband outputs.
        quadToComplex(commandQueue, levelTemps.lohi, 
                      *subbands, 2, 3,
                      {levelTemps.lohiDone}, &(*events)[0]); 

        quadToComplex(commandQueue, levelTemps.hilo, 
                      *subbands, 0, 5,
                      {levelTemps.hiloDone}, &(*events)[1]); 

        quadToComplex(commandQueue, levelTemps.bpbp, 
                      *subbands, 1, 4,
                      {levelTemps.bpbpDone}, &(*events)[2]); 
 
    }
}


void Dtcwt::decimateFilter(cl::CommandQueue& commandQueue,
                           ImageBuffer<cl_float>& xx, 
                           const std::vector<cl::Event>& xxEvents,
                           LevelTemps& levelTemps, 
                           Subbands* subbands,
                           std::vector<cl::Event>* events)
{
    // Events are the events which, when done, signal that the Subband
    // outputs are complete

    // Definitely need to do this padding, whether outputs or not
    cl::Event xxPadded;
    padX(commandQueue, xx, xxEvents, &xxPadded);

    if (subbands == nullptr) {

        // Apply the non-decimating, low-pass filters both ways
        h0bx(commandQueue, xx, levelTemps.lo, 
             {xxPadded}, &levelTemps.loDone);

        cl::Event loPadded;
        padY(commandQueue, levelTemps.lo, {levelTemps.loDone}, &loPadded);
        h0by(commandQueue, levelTemps.lo, levelTemps.lolo,
             {loPadded}, &levelTemps.loloDone);


    } else {
        // If we've been given subbands to output to, we need to do more work:

        // Produce all the vertically-filtered versions
        h021bx(commandQueue, xx, levelTemps.lo,
               {xxPadded}, &levelTemps.loDone);

        // Create events that, when all done signify everything about this stage
        // is complete
        *events = std::vector<cl::Event>(1);

        cl::Event loPadded, hiPadded, bpPadded;
        padY(commandQueue, levelTemps.lo, {levelTemps.loDone}, &loPadded);
        padY(commandQueue, levelTemps.bp, {levelTemps.loDone}, &bpPadded);
        padY(commandQueue, levelTemps.hi, {levelTemps.loDone}, &hiPadded);

        // Prepare low-low output
        h0by(commandQueue, levelTemps.lo, levelTemps.lolo,
             {loPadded}, &levelTemps.loloDone);

        // ...and filter in the y direction, generating subband outputs.
        q2c_h1_h2_h0(commandQueue, levelTemps.lo, *subbands,
                     {loPadded, bpPadded, hiPadded},
                     &(*events)[0]);
     
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







