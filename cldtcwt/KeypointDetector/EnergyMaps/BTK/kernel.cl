// Copyright (C) 2013 Timothy Gale
__kernel void energyMap(const __global float2* sb,
                        const unsigned int sbStart,
                        const unsigned int sbPitch,
                        const unsigned int sbStride,
                        const unsigned int sbPadding,
                        const unsigned int sbWidth,
                        const unsigned int sbHeight,
                        __write_only image2d_t out
                        /*__global float* out,
                        const unsigned int outStride,
                        const unsigned int outPadding,
                        const unsigned int outWidth,
                        const unsigned int outHeight*/)
{
    int2 pos = (int2) (get_global_id(0), get_global_id(1));

    if (all(pos < (int2)(sbWidth, sbHeight))) {
    
        size_t idx = sbStart + pos.x + pos.y * sbStride;

        float minAbsH2 = INFINITY;
        for (int n = 0; n < 6; ++n) {
        
            // Sample the subband
            float2 h = sb[idx + sbPitch * n];

            // Convert to absolute (still squared, because it's more
            // convenient)
            minAbsH2 = fmin(minAbsH2, dot(h, h));

        }

        // Produce output
        write_imagef(out, pos, 2 * sqrt(minAbsH2));
        /*out[outPadding + pos.x + outStride * (outPadding + pos.y)]
               = result;*/

    }

}


