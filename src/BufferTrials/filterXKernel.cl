
#define FILTER_OFFSET ((FILTER_LENGTH-1) >> 1)


inline int wrap(int pos, int width)
{
    return min(max(pos, -1 - pos), width - pos);
}


// Working group width and height should be defined as WG_W and WG_H;
// the length of the filter is FILTER_LENGTH
__kernel
__attribute__((reqd_work_group_size(WG_W, WG_H, 1)))
void filterX(__global const float* input,
             __global float* output,
             unsigned int width, unsigned int stride,
             unsigned int height)
{
    int2 g = (int2) (get_global_id(0), get_global_id(1));
    int2 l = (int2) (get_local_id(0), get_local_id(1));

    __local float cache[WG_H][WG_W+FILTER_LENGTH-1];

    int px = wrap(g.x - FILTER_OFFSET, width);
    cache[l.y][l.x] = input[g.y*stride + px];

    if (l.x < (FILTER_LENGTH-1))
        cache[l.y][l.x+WG_W] 
            = input[g.y*stride + wrap(g.x+WG_W-FILTER_OFFSET, width)];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (g.x < width && g.y < height)
        output[g.y*stride + g.x] = cache[l.y][l.x];
}

