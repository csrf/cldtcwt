// Parameters: WG_SIZE_X
__kernel void findMax(read_only image2d_t input,
                      const float inputScale,
                      read_only image2d_t inFiner,
                      const float finerScale,
                      read_only image2d_t inCoarser,
                      const float coarserScale,
                      const float threshold,
                      global float2* maxCoords,
                      global int* numOutputs,
                      const int maxNumOutputs)
{
    sampler_t isNorm =
        CLK_NORMALIZED_COORDS_FALSE
        | CLK_ADDRESS_MIRRORED_REPEAT
        | CLK_FILTER_NEAREST;

    // Include extra one on each side to find whether edges are
    // maxima
    __local float inputLocal[1+WG_SIZE_Y+1]
                            [1+WG_SIZE_X+1];

    const int gx = get_global_id(0),
              gy = get_global_id(1),
              lx = get_local_id(0),
              ly = get_local_id(1);

    // Load in the complete region, with its border
    inputLocal[ly][lx] = read_imagef(input, isNorm, (float2) (gx-1, gy-1)).x;


    if (lx < 2)
        inputLocal[ly][lx+WG_SIZE_X]
            = read_imagef(input, isNorm, (float2) (gx-1 + WG_SIZE_X, gy-1)).x;

    if (ly < 2)
        inputLocal[ly+WG_SIZE_Y][lx]
            = read_imagef(input, isNorm,
                  (float2) (gx-1, gy-1 + " << wgSizeY_ << ")).x;

    if (lx < 2 && ly < 2)
        inputLocal[ly+" << wgSizeY_ << "][lx+" << wgSizeX_ << "]
            = read_imagef(input, isNorm,
  (float2) (gx-1+" << wgSizeX_ << ", gy-1 + " << wgSizeY_ << ")).x;

    // No need to do anything further if we're outside the image's
    // boundary
    if (gx >= get_image_width(input)
     || gy >= get_image_height(input))
        return;

    // Consider each of the surrounds; must be at least threshold,
    // anyway
    float surroundMax = threshold;
    surroundMax = max(surroundMax, inputLocal[ly  ][lx  ]);
    surroundMax = max(surroundMax, inputLocal[ly+1][lx  ]);
    surroundMax = max(surroundMax, inputLocal[ly+2][lx  ]);
    surroundMax = max(surroundMax, inputLocal[ly  ][lx+1]);
    surroundMax = max(surroundMax, inputLocal[ly+2][lx+1]);
    surroundMax = max(surroundMax, inputLocal[ly  ][lx+2]);
    surroundMax = max(surroundMax, inputLocal[ly+1][lx+2]);
    surroundMax = max(surroundMax, inputLocal[ly+2][lx+2]);

    if (inputLocal[ly+1][lx+1] > surroundMax) {
        
        // Now refine the position.  Not so refined as original
        // version: we ignore the cross term between x and y (since
        // they would involve pseudo-inverses)
        float ratioX = 
               (inputLocal[ly+1][lx+2] - inputLocal[ly+1][lx+1])
             / (inputLocal[ly+1][lx  ] - inputLocal[ly+1][lx+1]);

        float xOut = 0.5f * (1-ratioX) / (1+ratioX) + (float) gx;

        float ratioY = 
               (inputLocal[ly+2][lx+1] - inputLocal[ly+1][lx+1])
             / (inputLocal[ly  ][lx+1] - inputLocal[ly+1][lx+1]);

        float yOut = 0.5f * (1-ratioY) / (1+ratioY) + (float) gy;

        // Check levels up and level down.  Conveniently (in a way)
        // the centres of the images remain the centre from level to
        // level.

        // Get positions relative to centre
        float xc = xOut - 0.5f * (get_image_width(input) - 1.0f);
        float yc = xOut - 0.5f * (get_image_height(input) - 1.0f);

        // Check the level coarser
        float2 coarserCoords = (float2)
           (xc / 2.0f + 0.5f * (get_image_width(inCoarser) - 1.0f),
            yc / 2.0f + 0.5f * (get_image_height(inCoarser) - 1.0f));

        float coarserVal = read_imagef(inCoarser, isNorm,
                                       coarserCoords).x;

        // Check the level finer
        float2 finerCoords = (float2)
           (xc * 2.0f + 0.5f * (get_image_width(inFiner) - 1.0f),
            yc * 2.0f + 0.5f * (get_image_height(inFiner) - 1.0f));

        float finerVal = read_imagef(inFiner, isNorm,
                                     finerCoords).x;

        // We also need the current level at the max point
        float inputVal = read_imagef(input, isNorm,
                                     (float2) (xOut, yOut)).x;

        if (inputVal > coarserVal && inputVal > finerVal) {

            int ourOutputPos = atomic_inc(numOutputs);

            // Write it out (if there's enough space)
            if (ourOutputPos < maxNumOutputs)
                maxCoords[ourOutputPos] = (float2) (xOut, yOut);

        }

     }
}


