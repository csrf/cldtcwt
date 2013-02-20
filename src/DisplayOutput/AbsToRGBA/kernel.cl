__kernel
void absToRGBA(const __global float2* input,
               unsigned int padding,
               unsigned int stride,
               __write_only image2d_t output,
               float gain)
{
    int2 pos = (int2) (get_global_id(0), get_global_id(1));

    // Make sure we're within the valid region
    if (all(pos < get_image_dim(output))) {

        float v = gain * 
            fast_length(input[padding + pos.x + stride * (padding + pos.y)]);
        write_imagef(output, pos, (float4) (v, v, v, 1.0f));

    }
}

