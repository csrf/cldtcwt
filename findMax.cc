#include "findMax.h"

#include <string>
#include <sstream>
#include <iostream>

#include <stdexcept>

static const std::string acquireLockFn =
    "void acquireLock(__global volatile int* lock)"
    "{"
        // Wait until the lock was free before a swap
        "while (atomic_cmpxchg(lock, 0, 1) == 1)"
            ";"
    "}\n";


static const std::string releaseLockFn =
    "void releaseLock(__global volatile int* lock)"
    "{"
        // Unlock
        "atomic_xchg(lock, 0);"
    "}\n";


FindMax::FindMax(cl::Context& context,
               const std::vector<cl::Device>& devices)
   : context_(context), 
     wgSizeX_(16), wgSizeY_(16)
{
    // The OpenCL kernel:
    std::ostringstream kernelInput;

    // We'll need to use locking functions
    kernelInput << acquireLockFn
                << releaseLockFn;

    kernelInput
    << "__kernel void findMax(read_only image2d_t input,"
                             "read_only image2d_t inFiner,"
                             "read_only image2d_t inCoarser,"
                             "const float threshold,"
                             "global float2* maxCoords,"
                             "global int* numOutputs,"
                             "const int maxNumOutputs,"
                             "volatile global int* lock)\n"
        "{"
            "sampler_t isNorm ="
                "CLK_NORMALIZED_COORDS_FALSE"
                "| CLK_ADDRESS_MIRRORED_REPEAT"
                "| CLK_FILTER_NEAREST;\n"

            // Include extra one on each side to find whether edges are
            // maxima
            "__local float inputLocal[1+" << wgSizeY_ << "+1]"
                                    "[1+" << wgSizeX_ << "+1];\n"

            "const int gx = get_global_id(0),"
                      "gy = get_global_id(1),"
                      "lx = get_local_id(0),"
                      "ly = get_local_id(1);\n"

            // Load in the complete region, with its border
            "inputLocal[ly][lx] = read_imagef(input, isNorm,"
                                    "(float2) (gx-1, gy-1)).x;\n"

            "if (lx < 2)"
                "inputLocal[ly][lx+" << wgSizeX_ << "]"
                    "= read_imagef(input, isNorm,"
                          "(float2) (gx-1 + " << wgSizeX_ << ", gy-1)).x;\n"

            "if (ly < 2)"
                "inputLocal[ly+" << wgSizeY_ << "][lx]"
                    "= read_imagef(input, isNorm,"
                          "(float2) (gx-1, gy-1 + " << wgSizeY_ << ")).x;"

            "if (lx < 2 && ly < 2)"
                "inputLocal[ly+" << wgSizeY_ << "][lx+" << wgSizeX_ << "]"
                    "= read_imagef(input, isNorm,"
          "(float2) (gx-1+" << wgSizeX_ << ", gy-1 + " << wgSizeY_ << ")).x;"

            // No need to do anything further if we're outside the image's
            // boundary
            "if (gx >= get_image_width(input)"
             "|| gy >= get_image_height(input))"
                "return;"

            // Consider each of the surrounds; must be at least threshold,
            // anyway
            "float surroundMax = threshold;"
            "surroundMax = max(surroundMax, inputLocal[ly  ][lx  ]);"
            "surroundMax = max(surroundMax, inputLocal[ly+1][lx  ]);"
            "surroundMax = max(surroundMax, inputLocal[ly+2][lx  ]);"
            "surroundMax = max(surroundMax, inputLocal[ly  ][lx+1]);"
            "surroundMax = max(surroundMax, inputLocal[ly+2][lx+1]);"
            "surroundMax = max(surroundMax, inputLocal[ly  ][lx+2]);"
            "surroundMax = max(surroundMax, inputLocal[ly+1][lx+2]);"
            "surroundMax = max(surroundMax, inputLocal[ly+2][lx+2]);"

            "if (inputLocal[ly+1][lx+1] > surroundMax) {"
                
                // Now refine the position.  Not so refined as original
                // version: we ignore the cross term between x and y (since
                // they would involve pseudo-inverses)
                "float ratioX = "
                       "(inputLocal[ly+1][lx+2] - inputLocal[ly+1][lx+1])"
                     "/ (inputLocal[ly+1][lx  ] - inputLocal[ly+1][lx+1]);"

                "float xOut = 0.5f * (1-ratioX) / (1+ratioX) + (float) gx;"

                "float ratioY = "
                       "(inputLocal[ly+2][lx+1] - inputLocal[ly+1][lx+1])"
                     "/ (inputLocal[ly  ][lx+1] - inputLocal[ly+1][lx+1]);"

                "float yOut = 0.5f * (1-ratioY) / (1+ratioY) + (float) gy;"

                // Check levels up and level down.  Conveniently (in a way)
                // the centres of the images remain the centre from level to
                // level.

                // Get positions relative to centre
                "float xc = xOut - 0.5f * (get_image_width(input) - 1.0f);"
                "float yc = xOut - 0.5f * (get_image_height(input) - 1.0f);"

                // Check the level coarser
                "float2 coarserCoords = (float2)"
                   "(xc / 2.0f + 0.5f * (get_image_width(inCoarser) - 1.0f),"
                   " yc / 2.0f + 0.5f * (get_image_height(inCoarser) - 1.0f));"

                "float coarserVal = read_imagef(inCoarser, isNorm,"
                                               "coarserCoords).x;"

                // Check the level finer
                "float2 finerCoords = (float2)"
                   "(xc * 2.0f + 0.5f * (get_image_width(inFiner) - 1.0f),"
                   " yc * 2.0f + 0.5f * (get_image_height(inFiner) - 1.0f));"

                "float finerVal = read_imagef(inFiner, isNorm,"
                                             "finerCoords).x;"

                // We also need the current level at the max point
                "float inputVal = read_imagef(input, isNorm,"
                                             "(float2) (xOut, yOut)).x;"

                "if (inputVal > coarserVal && inputVal > finerVal) {"

                    "int ourOutputPos = atomic_inc(numOutputs);"

                    // Write it out (if there's enough space)
                    "if (ourOutputPos < maxNumOutputs)"
                        "maxCoords[ourOutputPos] = (float2) (xOut, yOut);"

                "}"

            "}"
        "}";

    const std::string sourceCode = kernelInput.str();

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
    kernel_ = cl::Kernel(program, "findMax");
}

static int roundWGs(int l, int lWG)
{
    return lWG * (l / lWG + ((l % lWG) ? 1 : 0)); 
}

void FindMax::operator() 
      (cl::CommandQueue& commandQueue,
       const cl::Image2D& input,
       const cl::Image2D& inputFiner,
       const cl::Image2D& inputCoarser,
       float threshold,
       cl::Buffer& output,
       cl::Buffer& numOutputs,
       cl::Buffer& lock,
       const std::vector<cl::Event>& waitEvents,
       cl::Event* doneEvent)
{
    // Run the filter for each location in output (which determines
    // the locations to run at) using commandQueue.  input and output are
    // both single-component float images.  filter is a vector of floats.
    // The command will not start until all of waitEvents have completed, and
    // once done will flag doneEvent.

    cl::NDRange WorkgroupSize = {wgSizeX_, wgSizeY_};

    cl::NDRange GlobalSize = {
        roundWGs(input.getImageInfo<CL_IMAGE_WIDTH>(), wgSizeX_), 
        roundWGs(input.getImageInfo<CL_IMAGE_HEIGHT>(), wgSizeY_)
    }; 

    // Set all the arguments
    kernel_.setArg(0, input);
    kernel_.setArg(1, inputFiner);
    kernel_.setArg(2, inputCoarser);
    kernel_.setArg(3, (threshold));
    kernel_.setArg(4, output);
    kernel_.setArg(5, numOutputs);
    kernel_.setArg(6, int(output.getInfo<CL_MEM_SIZE>() / (2 * sizeof(int))));
    kernel_.setArg(7, lock);

    // Execute
    commandQueue.enqueueNDRangeKernel(kernel_, cl::NullRange,
                                      GlobalSize,
                                      WorkgroupSize,
                                      &waitEvents, doneEvent);
}



