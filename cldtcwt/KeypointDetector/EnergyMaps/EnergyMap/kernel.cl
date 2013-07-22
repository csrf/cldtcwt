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

        float abs_h_2[6];

        float energy = 0.f;
        for (int n = 0; n < 6; ++n) {
        
            // Sample the subband
            float2 h = sb[idx + sbPitch * n];

            // Convert to absolute (still squared, because it's more
            // convenient)
            abs_h_2[n] = dot(h, h);

            energy += abs_h_2[n];

        }

        float e = sqrt(energy);

        // Calculate result
        float result =
            (  sqrt(abs_h_2[0] * abs_h_2[3]) 
             + sqrt(abs_h_2[1] * abs_h_2[4]) 
             + sqrt(abs_h_2[2] * abs_h_2[5]))
            / fmax(1.e-6f, e)
                - 0.3f * e; 

        // Produce output
        write_imagef(out, pos, 4.f * result);
        /*out[outPadding + pos.x + outStride * (outPadding + pos.y)]
               = result;*/

    }

}


