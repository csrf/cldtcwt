// Implementation of interleaved DTCWT

#include "intDtcwt.h"


IntDtcwt::IntDtcwt(cl::Context& context,

                   std::vector<cl::Device>& devices,
                   float scaleFactor)
    : context_{context},
      scaleImageToImageBuffer_{context, devices},
      dtcwt_{context, devices, scaleFactor}
{}



void IntDtcwt::operator() (cl::CommandQueue& cq,
                           cl::Image2D& image,
                           IntDtcwtOutput& output,
                           const std::vector<cl::Event>& waitEvents)
{
    for (size_t n = 0; n < output.scales_.size(); ++n) {

        cl::Event imageBufferReady;
        scaleImageToImageBuffer_(cq, 
                                 image, output.inputImages_[n], 
                                 output.scales_[n],
                                 waitEvents, &imageBufferReady);
        
        dtcwt_(cq, output.inputImages_[n], output.dtcwtTemps_[n], 
                   output.dtcwtOutputs_[n], 
                   {imageBufferReady});

    }

}



IntDtcwtOutput IntDtcwt::createOutputs(size_t width, size_t height, 
                                       size_t startLevel, size_t numLevels,
                                       const std::vector<float> &scales)
{
    IntDtcwtOutput output;

    // Straightforward copy of some parameters
    output.inputWidth_ = width;
    output.inputHeight_ = height;
    output.startLevel_ = startLevel;
    output.numLevels_ = numLevels;
    output.scales_ = scales;

    for (float s: output.scales_) {

        // Generate an ImageBuffer to scale into
        size_t w = s * width, h = s * height;

        // Create a place for the scaled image
        output.inputImages_.emplace_back(context_, CL_MEM_READ_WRITE,
                                         w, h, 16, 32);

        // Create the temporary for this tree
        output.dtcwtTemps_.emplace_back(context_, w, h,
                                        startLevel, numLevels);

        // Create the outputs
        output.dtcwtOutputs_
            .push_back(output.dtcwtTemps_.back().createOutputs());

    }

    return output;
}





// Get the number of trees
size_t IntDtcwtOutput::numTrees() const
{
    return scales_.size();
}

// Get the number of levels calculated per tree
size_t IntDtcwtOutput::numLevels() const
{
    return numLevels_;
}

// Get the number of the first level calculated
// in all the trees
size_t IntDtcwtOutput::startLevel() const
{
    return startLevel_;
}

// Get the index of the output
size_t IntDtcwtOutput::idxFromTreeLevel(size_t tree, size_t level) const
{
    return tree + numTrees() * (level - startLevel_);
}

// Convert from index to level and tree number
std::tuple<size_t,size_t> IntDtcwtOutput::treeLevelFromIdx(size_t idx) const
{
    size_t tree = idx % numTrees();
    size_t level = startLevel_ + (idx - tree) / numTrees();
    return std::make_tuple(tree, level); 
}

// Get the scale of the level number or index
float IntDtcwtOutput::scale(size_t tree, size_t level) const
{
    return scales_[tree] * (1 << level);
}

float IntDtcwtOutput::scale(size_t idx) const
{
    size_t tree, level;
    std::tie(tree, level) = treeLevelFromIdx(idx);
    return scale(tree, level);
}


// Get requested subbands by tree and tree level
Subbands& IntDtcwtOutput::level(size_t tree, size_t level)
{
    return dtcwtOutputs_[tree].level(level);
}


const Subbands& IntDtcwtOutput::level(size_t tree, size_t level) const
{
    return dtcwtOutputs_[tree].level(level);
}


// Get the subband by index size_to all the trees
Subbands& IntDtcwtOutput::operator[](size_t idx)
{
    size_t tree, level;
    std::tie(tree, level) = treeLevelFromIdx(idx);
    return dtcwtOutputs_[tree].level(level);
}

const Subbands& IntDtcwtOutput::operator[](size_t idx) const
{
    size_t tree, level;
    std::tie(tree, level) = treeLevelFromIdx(idx);
    return dtcwtOutputs_[tree].level(level);
}


std::vector<cl::Event> 
    IntDtcwtOutput::doneEvents(size_t tree, size_t level) const
{
    return dtcwtOutputs_[tree].doneEvents(level);
}


std::vector<cl::Event> IntDtcwtOutput::doneEvents(size_t idx) const
{
    size_t tree, level;
    std::tie(tree, level) = treeLevelFromIdx(idx);
    return doneEvents(tree, level);
}







