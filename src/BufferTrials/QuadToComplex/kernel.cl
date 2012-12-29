// PADDING should have been defined externally, as should WG_W
// and WG_H (width and height of the workgroup respectively)
__attribute__((reqd_work_group_size(WG_W, WG_H, 1)))
__kernel void quadToComplex(__global const float* input,
                            unsigned int stride,
                            __write_only image2d_t output0,
                            __write_only image2d_t output1)
{
    const int2 g = (int2) (get_global_id(0), get_global_id(1));
    const int2 l = (int2) (get_local_id(0), get_local_id(1));

    // Load the values to local to get best read performance
    __local float cache[WG_H][WG_W];

    cache[l.y][l.x] = input[(PADDING + g.y) * stride
                           + PADDING + g.x];

    int2 outPos = g >> 1;

    barrier(CLK_LOCAL_MEM_FENCE);

    // Output only using the top right of each square of four pixels,
    // and only within the confines of the image
    if (all(outPos < get_image_dim(output0)) && !(g.x & 1) && !(g.y & 1)) {

        // Sample upper left, upper right, etc
        float ul = cache[l.y][l.x];
        float ur = cache[l.y][l.x+1];
        float ll = cache[l.y+1][l.x];
        float lr = cache[l.y+1][l.x+1];

        const float factor = 1.0f / sqrt(2.0f);

        // Combine into complex pairs
        write_imagef(output0, outPos,
                     factor * (float4) (ul - lr, ur + ll, 0.0, 1.0));
        write_imagef(output1, outPos,
                     factor * (float4) (ul + lr, ur - ll, 0.0, 1.0));

    }

}


