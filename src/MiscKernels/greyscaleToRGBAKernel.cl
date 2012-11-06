__kernel
void greyscaleToRGBA(__read_only image2d_t input,
                     __write_only image2d_t output,
                     float gain)
{
    sampler_t s = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST;

    int2 pos = (int2) (get_global_id(0), get_global_id(1));

    // Make sure we're within the valid region
    if (all(pos < get_image_dim(output))) {

        float v = gain * read_imagef(input, s, pos).s0;
        write_imagef(output, pos, (float4) (v, v, v, 1.0f));

    }
}

