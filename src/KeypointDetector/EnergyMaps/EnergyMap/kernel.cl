__kernel void energyMap(const __global float2* sb0,
                        const __global float2* sb1,
                        const __global float2* sb2,
                        const __global float2* sb3,
                        const __global float2* sb4,
                        const __global float2* sb5,
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
    
        size_t sbIdx = sbPadding + pos.x 
                     + (sbPadding + pos.y) * sbStride;

        // Sample each subband
        float2 h0 = sb0[sbIdx];
        float2 h1 = sb1[sbIdx];
        float2 h2 = sb2[sbIdx];
        float2 h3 = sb3[sbIdx];
        float2 h4 = sb4[sbIdx];
        float2 h5 = sb5[sbIdx];

        // Convert to absolute (still squared, because it's more
        // convenient)
        float abs_h0_2 = dot(h0,h0);
        float abs_h1_2 = dot(h1,h1);
        float abs_h2_2 = dot(h2,h2);
        float abs_h3_2 = dot(h3,h3);
        float abs_h4_2 = dot(h4,h4);
        float abs_h5_2 = dot(h5,h5);

        // Calculate result
        float result =
            (  sqrt(abs_h0_2 * abs_h3_2) 
             + sqrt(abs_h1_2 * abs_h4_2) 
             + sqrt(abs_h2_2 * abs_h5_2))
            /
            sqrt(0.01 + 
               1.5 * (  abs_h0_2 + abs_h1_2 + abs_h2_2
                        + abs_h3_2 + abs_h4_2 + abs_h5_2)); 

        // Produce output
        write_imagef(out, pos, result);
        /*out[outPadding + pos.x + outStride * (outPadding + pos.y)]
               = result;*/

    }

}


