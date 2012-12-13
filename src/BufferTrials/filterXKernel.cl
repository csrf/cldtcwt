__kernel
void filterX(__global const float* input,
             __global float* output,
             unsigned int width, unsigned int stride,
             unsigned int height)
{
    int2 pos = (int2) (get_global_id(0), get_global_id(1));

    // Make sure we're within the valid region
    //if (all(pos < get_image_dim(output))) {

     //   float v = gain * read_imagef(input, s, pos).s0;
      //  write_imagef(output, pos, (float4) (v, v, v, 1.0f));

    //}
}

