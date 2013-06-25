

// Load a rectangular region from a floating-point image
void readImageRegionToShared(__read_only image2d_t input,
                sampler_t sampler,
                float2 regionStart,
                int2 regionSize, 
                __local volatile float* output)
{
    // Take a region of regionSize, and load it into local memory with the
    // while workgroup.  Memory is laid out reading along the direction of
    // x.

    // We'll extract a rectangle the size of a workgroup each time.

    // The position within the workgroup
    int2 localPos = (int2) (get_local_id(0), get_local_id(1));

    // Loop over the rectangles
    for (int x = 0; x < regionSize.x; x += get_local_size(0)) {
        for (int y = 0; y < regionSize.y; y += get_local_size(1)) {

            int2 readPosOffset = (int2) (x,y) + localPos;

            bool inRegion = all(readPosOffset < regionSize);

            // Make sure we are still in the rectangular region asked for
            if (inRegion)
                output[readPosOffset.y * regionSize.x + readPosOffset.x]
                    = read_imagef(input, sampler, 
                            regionStart + convert_float2(readPosOffset)).x;

        }
    }
}

typedef struct {
    float a0, ax, ay, ahalfxx, ahalfyy, axy;
} QuadraticCoeffs;


void solveQuadraticCoefficients(__private QuadraticCoeffs* coeffs,
                                __local const volatile float* row0,
                                __local const volatile float* row1,
                                __local const volatile float* row2)
{
    // Takes a 3x3 area (one value of y for each row), and fits a quadratic
    // surface.  Corners are given 1/4 the weight for fitting purposes.

    // Octave/MATLAB script to generate the pseudoinverse
    // [x, y] = ind2sub([3 3], (1:9)'); x = x-2; y = y-2;
    // 
    // % Values to multiply a_0, a_x, a_y, a_xx/2, a_yy/2, a_xy by
    // P = [ones(9,1), x, y, x.*x/2, y.*y/2, x.*y];
    // 
    // % Scaling so corners are weighted down
    // S = diag(2.^(2-abs(x)-abs(y)));
    // 
    // inverse = ((S*P)'*(S*P)) \ (S*P)' * S

    const float inverse[6][9] = 
    {
        {-0.027778,    0.055556,   -0.027778,    0.055556,     0.88889,    0.055556,   -0.027778,    0.055556,   -0.027778},
        {-0.083333,           0,    0.083333,    -0.33333,           0,     0.33333,   -0.083333,           0,    0.083333},
        {-0.083333,    -0.33333,   -0.083333,           0,           0,           0,    0.083333,     0.33333,    0.083333},
        {  0.16667,    -0.33333,     0.16667,     0.66667,     -1.3333,     0.66667,     0.16667,    -0.33333,     0.16667},
        {  0.16667,     0.66667,     0.16667,    -0.33333,     -1.3333,    -0.33333,     0.16667,     0.66667,     0.16667},
        {     0.25,           0,       -0.25,           0,           0,           0,       -0.25,           0,        0.25}
    };

    coeffs->a0 = 0;
    coeffs->ax = 0;
    coeffs->ay = 0;
    coeffs->ahalfxx = 0;
    coeffs->ahalfyy = 0;
    coeffs->axy = 0;

    for (size_t n = 0; n < 9; ++n) {

        float v;

        if (n < 3)
            v = row0[n];
        else if (n < 6)
            v = row1[n-3];
        else 
            v = row2[n-6];

        coeffs->a0 += v * inverse[0][n];
        coeffs->ax += v * inverse[1][n];
        coeffs->ay += v * inverse[2][n];
        coeffs->ahalfxx += v * inverse[3][n];
        coeffs->ahalfyy += v * inverse[4][n];
        coeffs->axy += v * inverse[5][n];
    }

}



// Parameters: WG_SIZE_X, WG_SIZE_Y need to be set for the work group size.
// POS_LEN should be the number of floats to make the output structure.
__kernel __attribute__((reqd_work_group_size(WG_SIZE_X, WG_SIZE_Y, 1)))
void findMax(__read_only image2d_t input,
             const float inputScale,

             __read_only image2d_t inFiner,
             const float finerScale,

             __read_only image2d_t inCoarser,
             const float coarserScale,

             const float threshold,
             const float eigenRatioThreshold,

             __write_only __global float* maxCoords,

             global volatile unsigned int* numOutputs,
             int numOutputsOffset,
             const int maxNumOutputs)
{
    // Scales are how many pixels there are in the original image for each
    // pixel in this image

    sampler_t sampler =
        CLK_NORMALIZED_COORDS_FALSE
        | CLK_ADDRESS_CLAMP_TO_EDGE
        | CLK_FILTER_LINEAR;
    
    // Note: use of linear filtering means we need to add a half to all,
    // to compensate for half subtracted in the filtering function.

    // Include extra one on each side to find whether edges are
    // maxima
    __local volatile float inputLocal[WG_SIZE_Y+2][WG_SIZE_X+2];

    const int2 g = (int2) (get_global_id(0), get_global_id(1)),
               l = (int2) (get_local_id(0), get_local_id(1));

    const float2 startCorner = (float2) 
        (get_group_id(0) * get_local_size(0) - 0.5f,
         get_group_id(1) * get_local_size(1) - 0.5f);

    // Load region, with a border of one all around
    readImageRegionToShared(input, sampler, 
                            startCorner, // Corner
                            (int2) (WG_SIZE_X+2, WG_SIZE_Y+2), // Size
                            &inputLocal[0][0]);

    // Make sure the load is entirely finished
    barrier(CLK_LOCAL_MEM_FENCE);

    // No need to do anything further if we're outside the image's boundary
    if (g.x >= get_image_width(input)
     || g.y >= get_image_height(input))
        return;

    // Consider each of the surrounds; must be at least threshold,
    // anyway
    float surroundMax = threshold;
    surroundMax = max(surroundMax, inputLocal[l.y+0][l.x+0]);
    surroundMax = max(surroundMax, inputLocal[l.y+1][l.x+0]);
    surroundMax = max(surroundMax, inputLocal[l.y+2][l.x+0]);
    surroundMax = max(surroundMax, inputLocal[l.y+0][l.x+1]);
    surroundMax = max(surroundMax, inputLocal[l.y+2][l.x+1]);
    surroundMax = max(surroundMax, inputLocal[l.y+0][l.x+2]);
    surroundMax = max(surroundMax, inputLocal[l.y+1][l.x+2]);
    surroundMax = max(surroundMax, inputLocal[l.y+2][l.x+2]);

    if (inputLocal[l.y+1][l.x+1] > surroundMax) {
        
        float2 inputCoords = (float2) ((float)g.x, (float)g.y);

        // Fit coefficients of a quadratic to the surface
        QuadraticCoeffs c;
        solveQuadraticCoefficients(&c,
                                   &inputLocal[l.y  ][l.x],
                                   &inputLocal[l.y+1][l.x],
                                   &inputLocal[l.y+2][l.x]);

        // Find the peak of the surface

        // Calculate the Hessian
        float det = 1.f / (c.ahalfxx * c.ahalfyy - c.axy * c.axy);

        float2 invhessian[2] = 
        {
            (float2) (det * c.ahalfyy,     det * -c.axy),
            (float2) (   det * -c.axy,  det * c.ahalfxx)
        };

        // Calculate the grad
        float2 grad = (float2) (c.ax, c.ay);

        float2 move = -(float2)(dot(invhessian[0], grad), dot(invhessian[1], grad));

        // Drop if the displacement suggests it should be elsewhere entirely
        if (any(fabs(move) > 1.f))
            return;

        inputCoords += move;

        // Check the eigenvalues of the Hessian of this fit to check that it
        // enough of a dot, rather than a line
#if 0
        float s = sqrt((c.axx - c.ayy) * (c.axx - c.ayy) + c.axy * c.axy);

        float l0t = fabs(c.axx + c.ayy - s);
        float l1t = fabs(c.axx + c.ayy + s);

        float lmin = min(l0t, l1t);
        float lmax = max(l0t, l1t);

        // Return if too edge-like
        if (lmin < (eigenRatioThreshold * lmax))
            return;
#endif

        // Output position relative to the centre of the image in the native
        // scaling
        float2 outPos = inputScale * 
            (inputCoords 
              - (float2) 0.5 
                * convert_float2(get_image_dim(input) - (int2) 1));


#ifdef CHECK_SCALE_MAX
        // Check levels up and level down.  Convenientl.y (in a way)
        // the centres of the images remain the centre from level to
        // level.

        // Check the level coarser
        float2 finerCoords = 
            outPos / finerScale 
            + (float2) 0.5f 
                * convert_float2(get_image_dim(inFiner) - (int2) 1);

        float2 coarserCoords = 
            outPos / coarserScale 
            + (float2) 0.5f
              * convert_float2(get_image_dim(inCoarser) - (int2) 1);

        float inputVal   = read_imagef(input, sampler, 
                                       inputCoords + (float2) 0.5f).s0;
        float finerVal   = read_imagef(inFiner, sampler, 
                                       finerCoords + (float2) 0.5f).s0;
        float coarserVal = read_imagef(inCoarser, sampler, 
                                       coarserCoords + (float2) 0.5f).s0;

        if ((inputVal > coarserVal) && (inputVal > finerVal)) {
#endif
            int ourOutputPos = atomic_inc(&numOutputs[numOutputsOffset]);

            // Write it out (if there's enough space)
            if (ourOutputPos < maxNumOutputs) {
                maxCoords[ourOutputPos*POS_LEN + 0] = outPos.x;
                maxCoords[ourOutputPos*POS_LEN + 1] = outPos.y;
                maxCoords[ourOutputPos*POS_LEN + 2] = inputScale;
            } else
                numOutputs[numOutputsOffset] = maxNumOutputs;

#ifdef CHECK_SCALE_MAX
        }
#endif
     }
}





