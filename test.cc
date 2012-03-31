#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

#include "filterer.h"
#include "clUtil.h"
#include "dtcwt.h"
#include <iomanip>

#include <stdexcept>
struct dtcwtFilters {
    cl::Buffer level1h0;
    cl::Buffer level1h1;
    cl::Buffer level1hbp;

    cl::Buffer level2h0;
    cl::Buffer level2h1;
    cl::Buffer level2hbp;
};

struct dtcwtKernels {
    cl::Kernel colDecimateFilter;
    cl::Kernel rowDecimateFilter;
    cl::Kernel colFilter;
    cl::Kernel rowFilter;
    cl::Kernel quadToComplex;
    cl::Kernel cornernessMap;
};

dtcwtKernels getKernels(cl::Program& program)
{
    dtcwtKernels kernels;

    // Turn these into kernels
    kernels.rowDecimateFilter = cl::Kernel(program, "rowDecimateFilter");
    kernels.colDecimateFilter = cl::Kernel(program, "colDecimateFilter");
    kernels.rowFilter = cl::Kernel(program, "rowFilter");
    kernels.colFilter = cl::Kernel(program, "colFilter");
    kernels.quadToComplex = cl::Kernel(program, "quadToComplex");
    kernels.cornernessMap = cl::Kernel(program, "cornernessMap");

    return kernels;
}
    


#if 0
void dtcwtTransform(cl::Context& context, cl::CommandQueue& commandQueue,
                    std::vector<std::vector<cl::Image2D> >& output,
                    cl::Image2D& input, dtcwtFilters& filters,
                    dtcwtKernels& kernels,
                    int numLevels, int startLevel = 1)
{
    //std::vector<std::vector<cl::Image2D> > output; 

    cl::Image2D lolo;

    // Go down the tree until the point where we need to start recording
    // the results
    for (int n = 1; n < startLevel; ++n) {

        if (n == 1) {

            int width = input.getImageInfo<CL_IMAGE_WIDTH>();
            int height = input.getImageInfo<CL_IMAGE_HEIGHT>();

            // Pad if an odd number of pixels
            bool padW = width & 1,
                 padH = height & 1;

            // Low-pass filter the rows...
            cl::Image2D lo = 
                createImage2D(context, width + padW, height);
            //rowFilter(context, commandQueue, kernels.rowFilter,
            //          lo, input, filters.level1h0);

            // ...and the columns
            lolo = 
                createImage2D(context, width + padW, height + padH);
            colFilter(context, commandQueue, kernels.colFilter,
                      lolo, lo, filters.level1h0);

        } else {

            int width = lolo.getImageInfo<CL_IMAGE_WIDTH>();
            int height = lolo.getImageInfo<CL_IMAGE_HEIGHT>();

            // Pad if a non-multiple of four
            bool padW = (width % 4) != 0,
                 padH = (height % 4) != 0;

            // Low-pass filter the rows...
            cl::Image2D lo = 
                createImage2D(context, width / 2 + padW, height);
            rowDecimateFilter(context, commandQueue, kernels.colDecimateFilter,
                              lo, lolo, filters.level2h0, padW);

            // ...and the columns
            lolo = 
                createImage2D(context, width / 2 + padW, height / 2 + padH);
            colDecimateFilter(context, commandQueue, kernels.colDecimateFilter,
                              lolo, lo, filters.level2h0, padH);
            
        }

    }

    // Transform the image
    for (int n = startLevel; n < (startLevel + numLevels); ++n) {
        cl::Image2D hilo, lohi, bpbp;

        if (n == 1) {

            int width = input.getImageInfo<CL_IMAGE_WIDTH>();
            int height = input.getImageInfo<CL_IMAGE_HEIGHT>();

            // Pad if an odd number of pixels
            bool padW = width & 1,
                 padH = height & 1;

            // Low (row) - high (cols)
            cl::Image2D lo = 
                createImage2D(context, width + padW, height);
            //rowFilter(context, commandQueue, kernels.rowFilter, 
            //          lo, input, filters.level1h0);

            
            lohi =
                createImage2D(context, width + padW, height + padH);
            colFilter(context, commandQueue, kernels.colFilter, 
                      lohi, lo, filters.level1h1);

            // High (row) - low (cols)
            cl::Image2D hi =
                createImage2D(context, width + padW, height);
            //rowFilter(context, commandQueue, kernels.rowFilter,
            //          hi, input, filters.level1h1);

            hilo =
                createImage2D(context, width + padW, height + padH);
            colFilter(context, commandQueue, kernels.colFilter,
                      hilo, hi, filters.level1h0);

            // Band pass - band pass
            cl::Image2D bp =
                createImage2D(context, width + padW, height);
            //rowFilter(context, commandQueue, kernels.rowFilter,
            //          bp, input, filters.level1hbp);

            bpbp =
                createImage2D(context, width + padW, height + padH);
            colFilter(context, commandQueue, kernels.colFilter,
                      bpbp, bp, filters.level1hbp);


            // Low - low
            lolo = 
                createImage2D(context, width + padW, height + padH);
            colFilter(context, commandQueue, kernels.colFilter,
                      lolo, lo, filters.level1h0);



        } else {

            int width = lolo.getImageInfo<CL_IMAGE_WIDTH>();
            int height = lolo.getImageInfo<CL_IMAGE_HEIGHT>();

            // Pad if an odd number of pixels
            bool padW = (width % 4) != 0,
                 padH = (height % 4) != 0;

            // Low (row) - high (cols)
            cl::Image2D lo = 
                createImage2D(context, width / 2 + padW, height);
            rowDecimateFilter(context, commandQueue, kernels.rowDecimateFilter,
                              lo, lolo, filters.level2h0, padW);

            lohi =
                createImage2D(context, width / 2 + padW, height / 2 + padH);
            colDecimateFilter(context, commandQueue, kernels.colDecimateFilter,
                              lohi, lo, filters.level2h1, padH);


            // High (row) - low (cols)
            cl::Image2D hi =
                createImage2D(context, width / 2 + padW, height);
            rowDecimateFilter(context, commandQueue, kernels.rowDecimateFilter,
                              hi, lolo, filters.level2h1, padW);

            hilo =
                createImage2D(context, width / 2 + padW, height / 2 + padH);
            colDecimateFilter(context, commandQueue, kernels.colDecimateFilter,
                              hilo, hi, filters.level2h0, padH);


            // Band pass - band pass
            cl::Image2D bp =
                createImage2D(context, width / 2 + padW, height);
            rowDecimateFilter(context, commandQueue, kernels.rowDecimateFilter,
                              bp, lolo, filters.level2hbp, padW);

            bpbp =
                createImage2D(context, width / 2 + padW, height / 2 + padH);
            colDecimateFilter(context, commandQueue, kernels.colDecimateFilter,
                              bpbp, bp, filters.level2hbp, padH);


            // Low - low
            lolo = 
                createImage2D(context, width / 2 + padW, height / 2 + padH);
            colDecimateFilter(context, commandQueue, kernels.colDecimateFilter,
                              lolo, lo, filters.level2h0, padH);

        }

        output.push_back(std::vector<cl::Image2D>());

        int idx = n - startLevel;
        int width = hilo.getImageInfo<CL_IMAGE_WIDTH>() / 2;
        int height = hilo.getImageInfo<CL_IMAGE_HEIGHT>() / 2;

        for (int n = 0; n < 12; ++n)
            output[idx].push_back(createImage2D(context, width, height));


        quadToComplex(context, commandQueue, kernels.quadToComplex,
                      output[idx][2], output[idx][2+6],
                      output[idx][3], output[idx][3+6],
                      lohi);

        quadToComplex(context, commandQueue, kernels.quadToComplex,
                      output[idx][0], output[idx][0+6],
                      output[idx][5], output[idx][5+6],
                      hilo);

        quadToComplex(context, commandQueue, kernels.quadToComplex,
                      output[idx][4], output[idx][4+6],
                      output[idx][1], output[idx][1+6],
                      bpbp);

    }
    //return output;

}
#endif


std::tuple<Filters, Filters>
        createFilters(cl::Context& context, cl::CommandQueue& commandQueue)
{
    Filters level1, level2;

    level1.h0 = createBuffer(context, commandQueue,
           { -0.0018, 0, 0.0223, -0.0469, -0.0482, 0.2969, 0.5555, 0.2969,
             -0.0482, -0.0469, 0.0223, 0, -0.0018} );

    level1.h1 = createBuffer(context, commandQueue, 
           { -0.0001, 0, 0.0013, -0.0019, -0.0072, 0.0239, 0.0556, -0.0517,
             -0.2998, 0.5594, -0.2998, -0.0517, 0.0556, 0.0239, -0.0072,
             -0.0019, 0.0013, 0, -0.0001 } );
    
    level1.hbp = createBuffer(context, commandQueue, 
           { -0.0004, -0.0006, -0.0001, 0.0042, 0.0082, -0.0074, -0.0615,
             -0.1482, -0.1171, 0.6529, -0.1171, -0.1482, -0.0615, -0.0074, 
             0.0082, 0.0042, -0.0001, -0.0006, -0.0004 } );

    level2.h0 = createBuffer(context, commandQueue, 
           { -0.0046, -0.0054, 0.0170, 0.0238, -0.1067, 0.0119, 0.5688,
             0.7561, 0.2753, -0.1172, -0.0389, 0.0347, -0.0039, 0.0033 } );

    level2.h1 = createBuffer(context, commandQueue, 
           { -0.0033, -0.0039, -0.0347, -0.0389, 0.1172, 0.2753, -0.7561,
             0.5688, -0.0119, -0.1067, -0.0238, 0.0170, 0.0054, -0.0046 } );

    level2.hbp = createBuffer(context, commandQueue, 
           { -0.0028, -0.0004, 0.0210, 0.0614, 0.1732, -0.0448, -0.8381,
             0.4368, 0.2627, -0.0076, -0.0264, -0.0255, -0.0096, -0.0000 } );

    return std::make_tuple(level1, level2);
}



void displayComplexImage(cl::CommandQueue& cq, cl::Image2D& image)
{
    const size_t width = image.getImageInfo<CL_IMAGE_WIDTH>(),
                height = image.getImageInfo<CL_IMAGE_HEIGHT>();
    float output[height][width][2];
    readImage2D(cq, &output[0][0][0], image);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x)
            std::cout << output[y][x][0] 
                      << "+i*" << output[y][x][1]<< "\t";

        std::cout << std::endl;
    }

    std::cout << std::endl;
}


int main()
{
    try {

        // Retrive platform information
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.size() == 0)
            throw std::runtime_error("No platforms!");

        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // Create a context to work in 
        cl::Context context(devices);


        // Ready the command queue on the first device to hand
        cl::CommandQueue commandQueue(context, devices[0]);

        Filters level1, level2;
        std::tie(level1, level2) = createFilters(context, commandQueue);

        int x = 0;
        const int numLevels = 2;
        const int startLevel = 1;


        //-----------------------------------------------------------------
        // Starting test code
        const size_t width = 8, height = 12;
        const size_t oWidth = 8, oHeight = 12;
  
        cl::Image2D inImage = createImage2D(context, width, height);
        float input[height][width] = {0.0f};
        for (int x = 4; x < width; ++x)
            for (int y = 0; y < height; ++y)
                input[y][x] = 1.0f;
        //input[4][2] = 1.0f;
        //
        //input[4][3] = -1.0f;

        writeImage2D(commandQueue, inImage, &input[0][0]);
        std::cout << "Creating Dtcwt" << std::endl;
        Dtcwt dtcwt(context, devices);

        std::cout << "Creating environment" << std::endl;
        DtcwtContext env = dtcwt.createContext(width, height,
                                               numLevels, startLevel,
                                               level1, level2);

        std::cout << "Running DTCWT" << std::endl;
        dtcwt(commandQueue, inImage, env);

        std::cout << "Displaying image" << std::endl;

        std::cout << std::setiosflags(std::ios_base::right
                        | std::ios_base::fixed)
                  << std::setprecision(2);

        for (auto& img: env.outputs[0])
            displayComplexImage(commandQueue, img);


    }
    catch (cl::Error err) {
        std::cerr << "Error: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
    }
                     
    return 0;
}



