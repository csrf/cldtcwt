__kernel
void pyramidSum(__read_only image2d_t input1, float gain1,
                __read_only image2d_t input2, float gain2,
                __write_only image2d_t output)
{
    sampler_t sNearest = CLK_NORMALIZED_COORDS_FALSE 
                       | CLK_FILTER_NEAREST
                       | CLK_ADDRESS_CLAMP_TO_EDGE;
    sampler_t sLinear = CLK_NORMALIZED_COORDS_FALSE 
                       | CLK_FILTER_LINEAR
                       | CLK_ADDRESS_CLAMP_TO_EDGE;

    int2 pos = (int2) (get_global_id(0), get_global_id(1));

    float2 centre1 = (convert_float2(get_image_dim(input1)) - (float2) 1)
                        / (float2) 2;

    float2 centre2 = (convert_float2(get_image_dim(input2)) - (float2) 1)
                        / (float2) 2;

    // Go to the position half the distance from the centre in input2
    float2 pos2 = (convert_float2(pos) - centre1) / (float2) 2
                 + centre2;

    // Make sure we're within the valid region
    if (all(pos < get_image_dim(output))) {

        float i1 = read_imagef(input1, sNearest, pos).s0;
        float i2 = read_imagef(input2, sLinear, pos2 + (float2) 0.5).s0;
                // Need to add on the 0.5 due to the way linear addressing 
                // subtracts it

        write_imagef(output, pos, gain1 * i1 + gain2 * i2);

    }
}

