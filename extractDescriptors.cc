
#include <sstream>



#include "extractDescriptors.h"
#include <iostream>
#include <iterator>
#include <string>
#include <algorithm>




// Function to calculate the cubic interpolation weights (see Keys 1981,
// Cubic Convolution Interpolation for Digital Image Processing).
static std::string kernelSrc = 
"void cubicCoefficients(float x, float coeffs[4])\n"
"{\n"
"    // x is between 0 and 1, and is the position of the point being\n"
"    // interpolated (minus the integer position).\n"
"    \n"
"    coeffs[0] = -0.5 * (x+1)*(x+1)*(x+1) + 2.5 * (x+1)*(x+1) - 4 * (x+1) + 2;\n"
"    coeffs[1] =  1.5 * (x  )*(x  )*(x  ) - 2.5 * (x  )*(x  )             + 1;\n"
"    coeffs[2] =  1.5 * (1-x)*(1-x)*(1-x) - 2.5 * (1-x)*(1-x)             + 1;\n"
"    coeffs[3] = -0.5 * (2-x)*(2-x)*(2-x) + 2.5 * (2-x)*(2-x) - 4 * (2-x) + 2;\n"
"}\n"
"\n"
"\n"
"    const sampler_t s = CLK_NORMALIZED_COORDS_FALSE\n"
"                      | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
"\n"
"\n"
"float2 readSBAndDerotate(__read_only image2d_t sb, float2 pos,\n"
"                        float2 angFreq, float2 offset)\n"
"{\n"
"    // Read pos, apply offset and derotate by angFreq\n"
"\n"
"    // Clamping means it returns zero out of the image, which is what we\n"
"    // want\n"
"\n"
"    float2 val = read_imagef(sb, s, pos).xy;\n"
"    // Apply offset to give consistent phase behaviour relative to sampling\n"
"    // point between subbands\n"
"    val = (float2) (val.x * offset.x - val.y * offset.y,\n"
"                    val.x * offset.y + val.y * offset.x);\n"
"\n"
"    // Find phase in each direction\n"
"    float phase = pos.x * angFreq.x + pos.y * angFreq.y;\n"
"\n"
"    // Find coefficients to multiply by\n"
"    float cosComp;\n"
"    float sinComp = sincos(-phase, &cosComp);\n"
"\n"
"    // Multiply and return\n"
"    return (float2) (cosComp * val.x - sinComp * val.y,\n"
"                     cosComp * val.y + sinComp * val.x);\n"
"\n"
"}\n"
"\n"
"\n"
"float2 centrePos(__read_only image2d_t img, float2 posRelToImgCentre)\n"
"{\n"
"    // Calculates the position relative to the top left hand corner, given\n"
"    // the position relative to the centre\n"
"\n"
"    // Convert coordinates from relative to the centre to relative to (0,0)\n"
"    float2 centre = ((float2) (get_image_width(img), get_image_height(img))\n"
"                    - 1.0f) / 2.0f;\n"
"\n"
"    return centre + posRelToImgCentre;\n"
"}\n"
"\n"
"\n"
"\n"
"__kernel void extractDescriptor(__read_only __global float2* pos,\n"
"                                int numPos,\n"
"                                __read_only __global float2* sampleLocs,\n"
"                                const int numSampleLocs,\n"
"                                int stride, int offset,\n"
"                                float scaleFactor,\n"
"                                __write_only __global float2* output,\n"
"                                __read_only image2d_t sb0,\n"
"                                __read_only image2d_t sb1,\n"
"                                __read_only image2d_t sb2,\n"
"                                __read_only image2d_t sb3,\n"
"                                __read_only image2d_t sb4,\n"
"                                __read_only image2d_t sb5)\n"
"{\n"
"\n"
"    // Complex numbers to subbands multiply by\n"
"    const float2 offsets[6] = {\n"
"        (float2) ( 0, 1), (float2) ( 0,-1), (float2) ( 0, 1), \n"
"        (float2) (-1, 0), (float2) ( 1, 0), (float2) (-1, 0)\n"
"    };\n"
"\n"
"    const float2 angularFreq[6] = {\n"
"        (float2) (-1,-3) * M_PI_F / 2.15f, \n"
"        (float2) (-sqrt(5.f), -sqrt(5.f)) * M_PI_F / 2.15f, \n"
"        (float2) (-3, -1) * M_PI_F / 2.15f, \n"
"        (float2) (-3,  1) * M_PI_F / 2.15f, \n"
"        (float2) (-sqrt(5.f), sqrt(5.f)) * M_PI_F / 2.15f, \n"
"        (float2) (-1, 3) * M_PI_F / 2.15f \n"
"    };\n"
"\n"
"\n"
"\n"
"    int idx = get_global_id(0);\n"
"    int xIdx = get_global_id(1);\n"
"    int yIdx = get_global_id(2);\n"
"\n"
"\n"
"    // Read coordinates from the input matrix\n"
"    float2 posFromCentre = pos[idx] * scaleFactor;\n"
"    \n"
"    // Calculate how far the keypoint is from the upper-left nearest pixel,\n"
"    // and the nearest lower integer location\n"
"    float2 intPos;\n"
"    float2 rounding = fract(centrePos(sb0, posFromCentre), \n"
"                            &intPos);\n"
"\n"
"    // Calculate the sampling position for this worker\n"
"    float2 samplePos = intPos + (float2) (xIdx, yIdx)\n"
"                     - DIAMETER / 2.0f - 1.0f;\n"
"\n"
"\n"
"\n"
"    // Storage for the subband values\n"
"    __local float2 sbVals[DIAMETER+4][DIAMETER+4];\n"
"\n"
"    float interpCoeffsX[4];\n"
"    float interpCoeffsY[4];\n"
"\n"
"    // Work out which sampling location we should take (if any)\n"
"    const int samplerIdx = xIdx + yIdx * get_local_size(1);\n"
"    const bool isSampler = samplerIdx < numSampleLocs;\n"
"    \n"
"    // The place where this worker picks its sample\n"
"    float2 samplerIntf;\n"
"    float2 samplerRound = fract(1.0 + DIAMETER / 2.0\n"
"                                + rounding + sampleLocs[samplerIdx],\n"
"                                &samplerIntf);\n"
"    int2 samplerInt = convert_int2(samplerIntf);\n"
"\n"
"    // Work item is one of those doing the sampling\n"
"    if (isSampler) {\n"
"        \n"
"        // Work out interpolation coefficient for current work item\n"
"        cubicCoefficients(samplerRound.x, interpCoeffsX);\n"
"        cubicCoefficients(samplerRound.y, interpCoeffsY);\n"
"        \n"
"    }\n"
"\n"
"\n"
"    // For each subband\n"
"    for (int n = 0; n < 6; ++n) {\n"
"\n"
"        // Select the correct subband as input\n"
"        switch (n) {\n"
"        case 0: \n"
"           sbVals[yIdx][xIdx] = readSBAndDerotate(sb0, samplePos, \n"
"                                           angularFreq[0], offsets[0]);\n"
"           break;\n"
"        case 1:\n"
"           sbVals[yIdx][xIdx] = readSBAndDerotate(sb1, samplePos, \n"
"                                           angularFreq[1], offsets[1]);\n"
"           break;\n"
"        case 2: \n"
"           sbVals[yIdx][xIdx] = readSBAndDerotate(sb2, samplePos, \n"
"                                           angularFreq[2], offsets[2]);\n"
"           break;\n"
"        case 3: \n"
"           sbVals[yIdx][xIdx] = readSBAndDerotate(sb3, samplePos, \n"
"                                           angularFreq[3], offsets[3]);\n"
"           break;\n"
"        case 4: \n"
"           sbVals[yIdx][xIdx] = readSBAndDerotate(sb4, samplePos, \n"
"                                           angularFreq[4], offsets[4]);\n"
"           break;\n"
"        case 5: \n"
"           sbVals[yIdx][xIdx] = readSBAndDerotate(sb5, samplePos, \n"
"                                           angularFreq[5], offsets[5]);\n"
"           break;\n"
"        }\n"
"\n"
"        \n"
"        // Make sure all items have got here\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"        // If we are one of sampling points, calculate the sample\n"
"        if (isSampler) {\n"
"\n"
"            // Interpolate\n"
"            float2 result = (float2) (0,0);\n"
"\n"
"            for (int i1 = 0; i1 < 4; ++i1) {\n"
"\n"
"                float2 tmp = (float2) (0,0);\n"
"\n"
"                for (int i2 = 0; i2 < 4; ++i2) {\n"
"                    tmp += interpCoeffsX[i2] \n"
"                        * sbVals[samplerInt.y - 1 + i1]\n"
"                                [samplerInt.x - 1 + i2];\n"
"\n"
"                }\n"
"\n"
"                result += interpCoeffsY[i1] * tmp;\n"
"\n"
"            }\n"
"\n"
"\n"
"            // Re-rotate\n"
"            float2 phases = \n"
"                (centrePos(sb0, posFromCentre) + sampleLocs[samplerIdx])\n"
"                * angularFreq[n];\n"
"\n"
"            float cosComp;\n"
"            float sinComp = sincos(phases.x + phases.y, &cosComp);\n"
"\n"
"            result = (float2) (result.x * cosComp - result.y * sinComp,\n"
"                               result.x * sinComp + result.y * cosComp);\n"
"\n"
"\n"
"            // Save to matrix\n"
"            output[n + samplerIdx * 6 + idx * stride * 6 + offset * 6]\n"
"                = result;\n"
"\n"
"        }\n"
"\n"
"        // Only move on when all local memory values are done being used\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    }\n"
"}\n"
"\n";





DescriptorExtracter::DescriptorExtracter(cl::Context& context,
                const std::vector<cl::Device>& devices,
                cl::CommandQueue& cq,
                const std::vector<float[2]>& samplingPattern,
                float scaleFactor,
                int outputStride, int outputOffset,
                int diameter)
 : context_(context), diameter_(diameter)
{
    // Define the diameter (total width/height of sampling pattern)
    // to begin with
    std::ostringstream kernelInput;

    kernelInput << "#define DIAMETER " << diameter << "\n"
                << kernelSrc;

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
                          &samplingPattern[0][0]);

    // ...and extract the useful part, viz the kernel
    kernel_ = cl::Kernel(program, "extractDescriptor");

    // Set arguments we already know
    kernel_.setArg(2, samplingPattern_);
    kernel_.setArg(3, int(samplingPattern.size()));
    kernel_.setArg(4, int(outputStride));
    kernel_.setArg(5, int(outputOffset));
    kernel_.setArg(6, float(scaleFactor));
}



void DescriptorExtracter::operator() 
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



