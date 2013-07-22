// Copyright (C) 2013 Timothy Gale
__kernel void energyMap(const __global float2* sb,
                        const unsigned int sbStart,
                        const unsigned int sbPitch,
                        const unsigned int sbStride,
                        const unsigned int sbPadding,
                        const unsigned int sbWidth,
                        const unsigned int sbHeight,
                        __write_only image2d_t out)
{
    int2 pos = (int2) (get_global_id(0), get_global_id(1));

    if (all(pos < (int2)(sbWidth, sbHeight))) {
    
        size_t idx = sbStart + pos.x + pos.y * sbStride;

        // Sample each subband
        float abs_h_2[6];
        for (int n = 0; n < 6; ++n) {
        
            // Sample the subband
            float2 h = sb[idx + sbPitch * n];

            // Convert to absolute (still squared, because it's more
            // convenient)
            abs_h_2[n] = dot(h, h);

        }
       
        // Approximate angular frequencies
        float wx[] = {-1.4612, -3.2674, -4.3836, -4.3836, -3.2674, -1.461};
        float wy[] = {-4.3836, -3.2674, -1.4612,  1.4612,  3.2674,  4.3836};

        float H00 = 0, H11 = 0, H01 = 0;

        for (int n = 0; n < 6; ++n) {
            H00 -= wx[n] * wx[n] * abs_h_2[n];
            H11 -= wy[n] * wy[n] * abs_h_2[n];
            H01 -= wx[n] * wy[n] * abs_h_2[n];
        }

        float root = sqrt(H00 * H00 + H11 * H11 - 2.f * H11 * H00
                          + 4.f * H01 * H01);

        float l0 = -(H00 + H11 + root) / 2.f;
        float l1 = -(H00 + H11 - root) / 2.f;


        // Calculate result
        float result = l0 * l0 / (l1 + 0.1f); 

        // Produce output
        write_imagef(out, pos, result);

    }

}


