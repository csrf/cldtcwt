// Function to calculate the cubic interpolation weights (see Keys 1981,
// Cubic Convolution Interpolation for Digital Image Processing).

void cubicCoefficients(float x, float coeffs[4])
{
    // x is between 0 and 1, and is the position of the point being
    // interpolated (minus the integer position).
    
    coeffs[0] = -0.5 * (x+1)*(x+1)*(x+1) + 2.5 * (x+1)*(x+1) - 4 * (x+1) + 2;
    coeffs[1] =  1.5 * (x  )*(x  )*(x  ) - 2.5 * (x  )*(x  )             + 1;
    coeffs[2] =  1.5 * (1-x)*(1-x)*(1-x) - 2.5 * (1-x)*(1-x)             + 1;
    coeffs[3] = -0.5 * (2-x)*(2-x)*(2-x) + 2.5 * (2-x)*(2-x) - 4 * (2-x) + 2;
}


    const sampler_t s = CLK_NORMALIZED_COORDS_FALSE
                      | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;


float2 readSBAndDerotate(__read_only image2d_t sb, float2 pos,
                        float2 angFreq, float2 offset)
{
    // Read pos, apply offset and derotate by angFreq

    // Clamping means it returns zero out of the image, which is what we
    // want

    float2 val = read_imagef(sb, s, pos).xy;
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


float2 centrePos(__read_only image2d_t img, float2 posRelToImgCentre)
{
    // Calculates the position relative to the top left hand corner, given
    // the position relative to the centre

    // Convert coordinates from relative to the centre to relative to (0,0)
    float2 centre = ((float2) (get_image_width(img), get_image_height(img))
                    - 1.0f) / 2.0f;

    return centre + posRelToImgCentre;
}



__kernel void extractDescriptor(__read_only __global float* pos,
                                float scale,
                                __read_only __global int* kpOffsets,
                                int kpOffsetsIdx,
                                __read_only __global float2* sampleLocs,
                                const int numSampleLocs,
                                int stride, int offset,
                                __write_only __global float2* output,
                                __read_only image2d_t sb0,
                                __read_only image2d_t sb1,
                                __read_only image2d_t sb2,
                                __read_only image2d_t sb3,
                                __read_only image2d_t sb4,
                                __read_only image2d_t sb5)
{

    // Complex numbers to subbands multiply by
    const float2 offsets[6] = {
        (float2) ( 0, 1), (float2) ( 0,-1), (float2) ( 0, 1), 
        (float2) (-1, 0), (float2) ( 1, 0), (float2) (-1, 0)
    };

    const float2 angularFreq[6] = {
        (float2) (-1,-3) * M_PI_F / 2.15f, 
        (float2) (-sqrt(5.f), -sqrt(5.f)) * M_PI_F / 2.15f, 
        (float2) (-3, -1) * M_PI_F / 2.15f, 
        (float2) (-3,  1) * M_PI_F / 2.15f, 
        (float2) (-sqrt(5.f), sqrt(5.f)) * M_PI_F / 2.15f, 
        (float2) (-1, 3) * M_PI_F / 2.15f 
    };



    int idx = get_global_id(0);
    int xIdx = get_global_id(1);
    int yIdx = get_global_id(2);

    size_t kpIdxsBegin = kpOffsets[kpOffsetsIdx],
           kpIdxsEnd = kpOffsets[kpOffsetsIdx+1];

    // Work out which input/output index we're working with
    size_t kpIdx = kpIdxsBegin + idx;

    // See whether we need to do anything
    if (kpIdx >= kpIdxsEnd)
        return;

    // Read coordinates from the input matrix
    float2 posFromCentre = (float2) (pos[NUM_FLOATS_PER_POS * kpIdx],
                                     pos[NUM_FLOATS_PER_POS * kpIdx + 1])
                            / (float2) scale;

    
    // Calculate how far the keypoint is from the upper-left nearest pixel,
    // and the nearest lower integer location
    float2 intPos;
    float2 rounding = fract(centrePos(sb0, posFromCentre), 
                            &intPos);

    // Calculate the sampling position for this worker
    float2 samplePos = intPos + (float2) (xIdx, yIdx)
                     - DIAMETER / 2.0f - 1.0f;



    // Storage for the subband values
    __local float2 sbVals[DIAMETER+4][DIAMETER+4];

    float interpCoeffsX[4];
    float interpCoeffsY[4];

    // Work out which sampling location we should take (if any)
    const int samplerIdx = xIdx + yIdx * get_local_size(1);
    const bool isSampler = samplerIdx < numSampleLocs;
    
    // The place where this worker picks its sample
    float2 samplerIntf;
    float2 samplerRound = fract(1.0 + DIAMETER / 2.0
                                + rounding + sampleLocs[samplerIdx],
                                &samplerIntf);
    int2 samplerInt = convert_int2(samplerIntf);

    // Work item is one of those doing the sampling
    if (isSampler) {
        
        // Work out interpolation coefficient for current work item
        cubicCoefficients(samplerRound.x, interpCoeffsX);
        cubicCoefficients(samplerRound.y, interpCoeffsY);
        
    }


    // For each subband
    for (int n = 0; n < 6; ++n) {

        // Select the correct subband as input
        switch (n) {
        case 0: 
           sbVals[yIdx][xIdx] = readSBAndDerotate(sb0, samplePos, 
                                           angularFreq[0], offsets[0]);
           break;
        case 1:
           sbVals[yIdx][xIdx] = readSBAndDerotate(sb1, samplePos, 
                                           angularFreq[1], offsets[1]);
           break;
        case 2: 
           sbVals[yIdx][xIdx] = readSBAndDerotate(sb2, samplePos, 
                                           angularFreq[2], offsets[2]);
           break;
        case 3: 
           sbVals[yIdx][xIdx] = readSBAndDerotate(sb3, samplePos, 
                                           angularFreq[3], offsets[3]);
           break;
        case 4: 
           sbVals[yIdx][xIdx] = readSBAndDerotate(sb4, samplePos, 
                                           angularFreq[4], offsets[4]);
           break;
        case 5: 
           sbVals[yIdx][xIdx] = readSBAndDerotate(sb5, samplePos, 
                                           angularFreq[5], offsets[5]);
           break;
        }

        
        // Make sure all items have got here
        barrier(CLK_LOCAL_MEM_FENCE);

        // If we are one of sampling points, calculate the sample
        if (isSampler) {

            // Interpolate
            float2 result = (float2) (0,0);

            for (int i1 = 0; i1 < 4; ++i1) {

                float2 tmp = (float2) (0,0);

                for (int i2 = 0; i2 < 4; ++i2) {
                    tmp += interpCoeffsX[i2] 
                        * sbVals[samplerInt.y - 1 + i1]
                                [samplerInt.x - 1 + i2];

                }

                result += interpCoeffsY[i1] * tmp;

            }


            // Re-rotate
            float2 phases = 
                (centrePos(sb0, posFromCentre) + sampleLocs[samplerIdx])
                * angularFreq[n];

            float cosComp;
            float sinComp = sincos(phases.x + phases.y, &cosComp);

            result = (float2) (result.x * cosComp - result.y * sinComp,
                               result.x * sinComp + result.y * cosComp);


            // Save to matrix
            output[n + samplerIdx * 6 + kpIdx * stride * 6 + offset * 6]
                = result;

        }

        // Only move on when all local memory values are done being used
        barrier(CLK_LOCAL_MEM_FENCE);

    }
}

