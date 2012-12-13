
// Working group width and height
#define WG_W 16
#define WG_H 16

__kernel
void filterX(__global const float* input,
             __global float* output,
             unsigned int width, unsigned int stride,
             unsigned int height)
{
    int2 g = (int2) (get_global_id(0), get_global_id(1));
    int2 l = (int2) (get_local_id(0), get_local_id(1));

    __local float cache[WG_W][WG_H];

    cache[l.y][l.x] = input[g.y*stride + g.x];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (g.x < width && g.y < height)
        output[g.y*stride + g.x] = cache[l.y][l.x];
}

