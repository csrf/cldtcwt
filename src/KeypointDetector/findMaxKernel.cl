
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
                            regionStart + readPosOffset).x;

        }
    }
}


// Parameters: WG_SIZE_X, WG_SIZE_Y need to be set
__kernel __attribute__((reqd_work_group_size(WG_SIZE_X, WG_SIZE_Y, 1)))
void findMax(__read_only image2d_t input,
             const float inputScale,

             read_only image2d_t inFiner,
             const float finerScale,

             read_only image2d_t inCoarser,
             const float coarserScale,

             const float threshold,

             global float2* maxCoords,

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

    // Load region, with a border of one all around
    readImageRegionToShared(input, sampler, 
                            convert_float2(g) - (float2) 0.5f, // Corner
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
        
        // Now refine the position.  Not so refined as original
        // version: we ignore the cross term between x and y (since
        // they would involve pseudo-inverses)
        /*float ratioX = 
               (inputLocal[l.y+1][l.x+2] - inputLocal[l.y+1][l.x+1])
             / (inputLocal[l.y+1][l.x  ] - inputLocal[l.y+1][l.x+1]);

        float xOut = 0.5f * (1-ratioX) / (1+ratioX) + (float) g.x;

        float ratioY = 
               (inputLocal[l.y+2][l.x+1] - inputLocal[l.y+1][l.x+1])
             / (inputLocal[l.y  ][l.x+1] - inputLocal[l.y+1][l.x+1]);

        float yOut = 0.5f * (1-ratioY) / (1+ratioY) + (float) g.y;*/

        float2 inputCoords = (float2) ((float)g.x, (float)g.y);


        // Output position relative to the centre of the image in the native
        // scaling
        float2 outPos = inputScale * 
            (inputCoords 
              - (float2) 0.5 
                * convert_float2(get_image_dim(input) - (int2) 1));


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

        if (inputVal > coarserVal && inputVal > finerVal) {

            int ourOutputPos = atomic_inc(&numOutputs[numOutputsOffset]);

            // Write it out (if there's enough space)
            if (ourOutputPos < maxNumOutputs)
                maxCoords[ourOutputPos] = outPos;
            else
                numOutputs[numOutputsOffset] = maxNumOutputs;

        }

     }
}





