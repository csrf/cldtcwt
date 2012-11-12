
#include <sstream>



#include "extractDescriptors.h"
#include <iostream>
#include <iterator>
#include <string>
#include <algorithm>


#include "KeypointDescriptor/extractDescriptorsKernel.h"


Interpolator::Interpolator(cl::Context& context,
                const std::vector<cl::Device>& devices,
                cl::CommandQueue& cq,
                const std::vector<Coord>& samplingPattern,
                float scaleFactor,
                int outputStride, int outputOffset,
                int diameter)
 : context_(context), diameter_(diameter)
{
    // Define the diameter (total width/height of sampling pattern)
    // to begin with
    std::ostringstream kernelInput;

    kernelInput << "#define DIAMETER (" << diameter << ")\n";

    // Get input from the source file
    const char* fileText = reinterpret_cast<const char*>
            (src_KeypointDescriptor_extractDescriptorsKernel_h_src);
    size_t fileTextLength = 
        src_KeypointDescriptor_extractDescriptorsKernel_h_src_len;

    std::copy(fileText, fileText + fileTextLength,
              std::ostream_iterator<char>(kernelInput));

    const std::string sourceCode = kernelInput.str();

    // Bundle the code up
    cl::Program::Sources source;
    source.push_back(std::make_pair(sourceCode.c_str(), 
                                    sourceCode.length()));

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

    // Upload the sampling pattern
    const size_t samplingPatternSize 
        = samplingPattern.size() * 2 * sizeof(float);

    samplingPattern_ = cl::Buffer(context_, CL_MEM_READ_ONLY, 
                                  samplingPatternSize);

    cq.enqueueWriteBuffer(samplingPattern_, CL_TRUE, 
                          0, samplingPatternSize,
                          &samplingPattern[0]);

    // ...and extract the useful part, viz the kernel
    kernel_ = cl::Kernel(program, "extractDescriptor");

    // Set arguments we already know
    kernel_.setArg(2, samplingPattern_);
    kernel_.setArg(3, int(samplingPattern.size()));
    kernel_.setArg(4, int(outputStride));
    kernel_.setArg(5, int(outputOffset));
    kernel_.setArg(6, float(scaleFactor));
}



void Interpolator::operator() 
               (cl::CommandQueue& cq,
                const LevelOutput& subbands,
                const cl::Buffer& locations,
                int numLocations,
                cl::Buffer& output,
                std::vector<cl::Event> waitEvents,
                cl::Event* doneEvent)
{
    // Set subband arguments
    kernel_.setArg(8, subbands.sb[0]);
    kernel_.setArg(9, subbands.sb[1]);
    kernel_.setArg(10, subbands.sb[2]);
    kernel_.setArg(11, subbands.sb[3]);
    kernel_.setArg(12, subbands.sb[4]);
    kernel_.setArg(13, subbands.sb[5]);

    // Set descriptor location arguments (which are scaled by scaleFactor,
    // set on creation), and are relative to the centre of the subband
    // images
    kernel_.setArg(0, locations);
    kernel_.setArg(1, int(numLocations));

    // Set output argument
    kernel_.setArg(7, output);

    // Wait for the subbands to be done too
    std::copy(subbands.done.begin(), subbands.done.end(),
              std::back_inserter(waitEvents));

    // Enqueue the kernel
    cl::NDRange workgroupSize = {1, diameter_+4, diameter_+4};
    cl::NDRange globalSize = {numLocations, diameter_+4, diameter_+4};

    cq.enqueueNDRangeKernel(kernel_, cl::NullRange,
                            globalSize, workgroupSize,
                            &subbands.done, doneEvent);    
}


// Keypoint extracter class

DescriptorExtracter::DescriptorExtracter
    (cl::Context& context, 
     const std::vector<cl::Device>& devices,
     cl::CommandQueue& cq)
{
    const float pi = 4 * atan(1);

    // Pattern of locations to sample at
    // Set up the centre
    std::vector<Coord> finePattern= {{0, 0}};

    // Set up the circle
    for (int n = 0; n < 12; ++n) {
        finePattern.push_back({float(sin(float(n) / 12.f * 2.f * pi)),
                               float(cos(float(n) / 12.f * 2.f * pi))});
    }

    std::vector<Coord> coarsePattern = {{0, 0}};

    // Set up the kernels
    fineInterpolator_ = Interpolator(context, devices, cq,
                                     finePattern, 1.0f, 14, 0, 2);
    coarseInterpolator_ = Interpolator(context, devices, cq,
                                     coarsePattern, 0.5f, 14, 13, 0);
}


void DescriptorExtracter:: operator() 
               (cl::CommandQueue& cq,
                const LevelOutput& fineSubbands,
                const LevelOutput& coarseSubbands,
                const cl::Buffer& locations,
                int numLocations,
                cl::Buffer& output,
                std::vector<cl::Event> waitEvents,
                cl::Event* doneEventFine, cl::Event* doneEventCoarse)
{
    // Call the fine and coarse levels
    fineInterpolator_(cq, fineSubbands, locations, numLocations, output,
                      waitEvents, doneEventFine);

    coarseInterpolator_(cq, coarseSubbands, locations, numLocations, output,
                      waitEvents, doneEventCoarse);
}


