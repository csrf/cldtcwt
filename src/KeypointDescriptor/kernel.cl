// Function to calculate the cubic interpolation weights (see Keys 1981,
// Cubic Convolution Interpolation for Digital Image Processing).


float2 readSBAndDerotate(const __global float2* sb, int2 pos,
                        float2 angFreq, float2 offset,
                        unsigned int padding, unsigned int stride)
{
    // Read pos, apply offset and derotate by angFreq

    // Clamping means it returns zero out of the image, which is what we
    // want
    float2 val = sb[padding + pos.x + (padding + pos.y) * stride];
    // Apply offset to give consistent phase behaviour relative to sampling
    // point between subbands
    val = (float2) (val.x * offset.x - val.y * offset.y,
                    val.x * offset.y + val.y * offset.x);

    // Find phase in each direction
    float phase = pos.x * angFreq.x + pos.y * angFreq.y;

    // Find coefficients to multiply by
    float cosComp;
    float sinComp = sincos(-phase, &cosComp);

    // Multiply and return
    return (float2) (cosComp * val.x - sinComp * val.y,
                     cosComp * val.y + sinComp * val.x);

}




void cubicCoefficients(float x, float coeffs[4])
{
    // x is between 0 and 1, and is the position of the point being
    // interpolated (minus the integer position).
    
    coeffs[0] = -0.5 * (x+1)*(x+1)*(x+1) + 2.5 * (x+1)*(x+1) - 4 * (x+1) + 2;
    coeffs[1] =  1.5 * (x  )*(x  )*(x  ) - 2.5 * (x  )*(x  )             + 1;
    coeffs[2] =  1.5 * (1-x)*(1-x)*(1-x) - 2.5 * (1-x)*(1-x)             + 1;
    coeffs[3] = -0.5 * (2-x)*(2-x)*(2-x) + 2.5 * (2-x)*(2-x) - 4 * (2-x) + 2;
}




float2 interp(const __local float2* values, unsigned int stride,
              int2 pos, 
              const float coeffsX[4],
              const float coeffsY[4]) 
{
    // Convolves the x and y filters with the values (a 2D array, with stride
    // length stride), starting the square at pos.
    float2 result = (float2) (0.f, 0.f);

    for (int iy = 0; iy < 4; ++iy) {

        float2 tmp = (float2) (0.f, 0.f);

        for (int ix = 0; ix < 4; ++ix) 
            tmp += coeffsX[ix] * values[stride * (pos.y + iy) +  pos.x + ix];

        result += coeffsY[iy] * tmp;
    }

    return result;
}





float2 rerotate(float2 dcValue, float2 pos, float2 angFreq)
{
    // Given location pos (relative to whereever the dcValue was derotated
    // from originally) and angular frequencies.

    // Calculate the complex phasor
    float c;
    float s = sincos(dot(pos, angFreq), &c);

    // Rotate by that phasor
    return (float2) (dcValue.x * c - dcValue.y * s,
                     dcValue.x * s + dcValue.y * c);
}




int2 ifract(float2 num, __private float2* fraction)
{
    // Convert a float2 to the integer part (returned)
    // and the remainder (fraction).
    float2 wholeNumf;
    *fraction = fract(num, &wholeNumf);
    return convert_int2(wholeNumf);
}







__kernel void extractDescriptor(const __global float* pos,
                                float scale,
                                const __global int* kpOffsets,
                                int kpOffsetsIdx,
                                const __global float2* sampleLocs,
                                const int numSampleLocs,
                                int stride, int offset,
                                __global float2* output,
                                const __global float2* sb0,
                                const __global float2* sb1,
                                const __global float2* sb2,
                                const __global float2* sb3,
                                const __global float2* sb4,
                                const __global float2* sb5,
                                unsigned int sbPadding,
                                unsigned int sbStride,
                                unsigned int sbWidth,
                                unsigned int sbHeight)
{

    // Complex numbers to subbands multiply by
    const float2 offsets[6] = {
        (float2) ( 0, 1), (float2) ( 0,-1), (float2) ( 0, 1), 
        (float2) (-1, 0), (float2) ( 1, 0), (float2) (-1, 0)
    };

    // Subband centre frequencies
    const float2 angularFreq[6] = {
        (float2) (-1,-3) * M_PI_F / 2.15f, 
        (float2) (-sqrt(5.f), -sqrt(5.f)) * M_PI_F / 2.15f, 
        (float2) (-3, -1) * M_PI_F / 2.15f, 
        (float2) (-3,  1) * M_PI_F / 2.15f, 
        (float2) (-sqrt(5.f), sqrt(5.f)) * M_PI_F / 2.15f, 
        (float2) (-1, 3) * M_PI_F / 2.15f 
    };




    size_t kpIdxsBegin = kpOffsets[kpOffsetsIdx],
           kpIdxsEnd = kpOffsets[kpOffsetsIdx+1];

    // Work out which input/output index we're working with
    const size_t kpIdx = kpIdxsBegin + get_global_id(0);

    // See whether we need to do anything
    if (kpIdx >= kpIdxsEnd)
        return;

    const int2 idx = (int2) (get_global_id(1), get_global_id(2));
    const int wgWidth = get_local_size(1);

    // Work out which sampling location we should take (if any)
    // (i.e. does this worker produce an output?)
    const int samplerIdx = idx.x + idx.y * wgWidth;
    const bool isSampler = samplerIdx < numSampleLocs;



    // Read coordinates from the input matrix
    float2 kpPos = (float2) (pos[NUM_FLOATS_PER_POS * kpIdx],
                             pos[NUM_FLOATS_PER_POS * kpIdx + 1])
                            / (float2) scale
                  + (float2) (sbWidth-1, sbHeight-1) / 2.f;

    
    // Calculate how far the keypoint is from the upper-left nearest pixel,
    // and the nearest lower integer location
    float2 kpRemPos;
    int2 kpIntPos = ifract(kpPos, &kpRemPos);



    // Calculate where this worker should be reading from.  The -1 at the 
    // end is to include enough area to do the interpolation properly.
    int2 readPos = kpIntPos + idx - (DIAMETER / 2) - 1;

    
    // The place where this worker picks its sample
    float2 sampleRemPosLocal;     
    int2 sampleIntPosLocal = ifract(1.0 + DIAMETER / 2.0
                               + kpRemPos + sampleLocs[samplerIdx],
                               &sampleRemPosLocal);


    // Work out interpolation coefficient for current work item
    float interpCoeffsX[4];
    cubicCoefficients(sampleRemPosLocal.x, interpCoeffsX);
    float interpCoeffsY[4];
    cubicCoefficients(sampleRemPosLocal.y, interpCoeffsY);
        

    // Storage for the subband values
    __local float2 sbVals[DIAMETER+4][DIAMETER+4];


    // For each subband
    for (int n = 0; n < 6; ++n) {

        const __global float2* sb;
        // Select the correct subband as input
        switch (n) {
        case 0: sb = sb0; break;
        case 1: sb = sb1; break;
        case 2: sb = sb2; break;
        case 3: sb = sb3; break;
        case 4: sb = sb4; break;
        case 5: sb = sb5; break;
        }

        sbVals[idx.y][idx.x]
                   = readSBAndDerotate(sb, readPos, 
                                       angularFreq[n], offsets[n],
                                       sbPadding, sbStride);

        // Make sure all items have got here
        barrier(CLK_LOCAL_MEM_FENCE);

        // If we are one of sampling points, sample
        if (isSampler) {

            // Interpolate and rerotate
            output[n + samplerIdx * 6 + kpIdx * stride * 6 + offset * 6]
              = rerotate(interp(&sbVals[0][0], wgWidth, 
                                sampleIntPosLocal - 1,
                                interpCoeffsX, interpCoeffsY),
                         kpPos + sampleLocs[samplerIdx],
                         angularFreq[n]);
        }

        // Only move on when all local memory values are done being used
        barrier(CLK_LOCAL_MEM_FENCE);

    }
}

