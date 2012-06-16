
#include <sstream>



// Function to calculate the cubic interpolation weights (see Keys 1981,
// Cubic Convolution Interpolation for Digital Image Processing).
static std::string cubicCoefficientsFn = 
"void cubicCoefficients(float x, float coeffs[4])"
"{
    // x is between 0 and 1, and is the position of the point being
    // interpolated (minus the integer position).
    
    "coeffs[0] = -0.5 * (x+1)*(x+1)*(x+1) + 2.5 * (x+1)*(x+1) - 4 * (x+1) + 2;"
    "coeffs[1] =  1.5 * (x  )*(x  )*(x  ) - 2.5 * (x  )*(x  )             + 1;"
    "coeffs[2] =  1.5 * (1-x)*(1-x)*(1-x) - 2.5 * (1-x)*(1-x)             + 1;"
    "coeffs[3] = -0.5 * (2-x)*(2-x)*(2-x) + 2.5 * (2-x)*(2-x) - 4 * (2-x) + 2;"
"}";



// Reference coordinates to the corner of the image (when they were
// previously relative to the centre)
float2 refToCorner(float2 coordsFromCentre, int width, int height)
{
    return coordsFromCentre + (float2) (width-1, height-1) / 2.0f;
}



void readImageBlock(image2d_t image, sampler_t sampler,
                    float2 corner, int width, int height)

// Function to read a width x height set of floats from an image, taking
// advantage of work groups to get the job done efficiently.  Blocks are
// loaded (at least conceptually) one at a time, starting with (x0,y0) in
// the top left hand corner, and moving down and right.  A local barrier at
// the end ensures everything is completed.  This last might be superfluous
// sometimes.
std::string readRegionFn(int stride, wgSizeX, wgSizeY)
{
"void readRegion(__read_only image2d_t input, sampler_t s,"
                "__local float* buffer,"
                "float x0, float y0,"
                "int width, int height)"
"{"
    "for (int x = get_local_id(0); x < width; x +=" << wgSizeX << ")"
        "for (int y = get_local_id(1); y < height; y += " << wgSizeX << ")"
            "buffer[x + y *" << stride << "]"
                "= read_imagef(input, s, (float2)(x0 + (float) x,"
                                                 "y0 + (float) y)).x;"

    // Make sure everything's done before moving on
    "barrier(CLK_LOCAL_MEM_FENCE);"
"}";
}
