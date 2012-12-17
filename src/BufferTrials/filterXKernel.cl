
#define FILTER_OFFSET ((FILTER_LENGTH-1) >> 1)
#define HALF_WG_W (WG_W >> 1)


inline int wrap(int pos, int width)
{
    return min(max(pos, -1 - pos), width - pos);
}

__constant float filter[FILTER_LENGTH];

// Working group width and height should be defined as WG_W and WG_H;
// the length of the filter is FILTER_LENGTH
__kernel
__attribute__((reqd_work_group_size(WG_W, WG_H, 1)))
void filterX(__global const float* input,
             __global float* output,
             unsigned int width, unsigned int stride,
             unsigned int height)
{
    const int2 g = (int2) (get_global_id(0), get_global_id(1));
    const int2 l = (int2) (get_local_id(0), get_local_id(1));

    const int pos = g.y*stride + g.x;

    __local float cache[WG_H][2*WG_W];

    // Load a rectangle two workgroups wide
    cache[l.y][l.x] = input[pos - HALF_WG_W];
    cache[l.y][l.x+WG_W] = input[pos + HALF_WG_W];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Wrap the ends, if needed.  Only one level of wrapping
    // is allowed, and the image must be at least HALF_WG_W
    // wide.
    if (g.x < (HALF_WG_W + ROW_PADDING))
        cache[l.y][l.x] = cache[l.y][WG_W - l.x - 1];

    const int upperOffset = g.x - ROW_PADDING + HALF_WG_W - width;
    if (upperOffset >= 0)
        cache[l.y][l.x+WG_W] = cache[l.y][l.x+WG_W
                                           - upperOffset - 1];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Calculate the convolution
    float v = 0.f;
    for (int n = 0; n < FILTER_LENGTH; ++n) 
         v = mad(cache[l.y][l.x + n + HALF_WG_W - FILTER_OFFSET], 
                 filter[n], v);        

    output[pos] = v;

}

