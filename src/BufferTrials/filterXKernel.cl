__kernel
void filterX(__global const float* input,
             __global float* output,
             unsigned int width, unsigned int stride,
             unsigned int height)
{
    int2 g = (int2) (get_global_id(0), get_global_id(1));

    if (g.x < width && g.y < height)
        output[g.y*stride + g.x] = input[g.y*stride + g.x];
}

