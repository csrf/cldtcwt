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
                             "global int2* maxCoords,"
                             "global int* numOutputs,"
                             "const int maxNumOutputs,"
                             "volatile global int* lock)\n"
        "{"
            "sampler_t inputSampler ="
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
            "inputLocal[ly][lx] = read_imagef(input, inputSampler,"
                                    "(int2) (gx-1, gy-1)).x;\n"

            "if (lx < 2)"
                "inputLocal[ly][lx+" << wgSizeX_ << "]"
                    "= read_imagef(input, inputSampler,"
                              "(int2) (gx-1 + " << wgSizeX_ << ", gy-1)).x;\n"

            "if (ly < 2)"
                "inputLocal[ly+" << wgSizeY_ << "][lx]"
                    "= read_imagef(input, inputSampler,"
                              "(int2) (gx-1, gy-1 + " << wgSizeY_ << ")).x;"

            "if (lx < 2 && ly < 2)"
                "inputLocal[ly+" << wgSizeY_ << "][lx+" << wgSizeX_ << "]"
                    "= read_imagef(input, inputSampler,"
              "(int2) (gx-1+" << wgSizeX_ << ", gy-1 + " << wgSizeY_ << ")).x;"

            // No need to do anything further if we're outside the image's
            // boundary
            "if (gx >= get_image_width(input)"
             "|| gy >= get_image_height(input))"
                "return;"

            // Consider each of the surrounds
            "float surroundMax =            inputLocal[ly  ][lx  ];"
            "surroundMax = max(surroundMax, inputLocal[ly+1][lx  ]);"
            "surroundMax = max(surroundMax, inputLocal[ly+2][lx  ]);"
            "surroundMax = max(surroundMax, inputLocal[ly  ][lx+1]);"
            "surroundMax = max(surroundMax, inputLocal[ly+2][lx+1]);"
            "surroundMax = max(surroundMax, inputLocal[ly  ][lx+2]);"
            "surroundMax = max(surroundMax, inputLocal[ly+1][lx+2]);"
            "surroundMax = max(surroundMax, inputLocal[ly+2][lx+2]);"

            "if (inputLocal[lx+1][ly+1] > surroundMax) {"
                "acquireLock(lock);"

                // Write it out (if there's enough space)
                //"if (*numOutputs < maxNumOutputs)"
                //    "maxCoords[(*numOutputs)++] = (int2) (gx, gy);"

                "releaseLock(lock);"
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
    kernel_.setArg(1, output);
    kernel_.setArg(2, numOutputs);
    kernel_.setArg(3, int(output.getInfo<CL_MEM_SIZE>() / (2 * sizeof(int))));
    kernel_.setArg(4, lock);

    // Execute
    commandQueue.enqueueNDRangeKernel(kernel_, cl::NullRange,
                                      GlobalSize,
                                      WorkgroupSize,
                                      &waitEvents, doneEvent);
}



