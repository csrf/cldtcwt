// Working group width and height should be defined as WG_W and WG_H;
// the length of the filter is FILTER_LENGTH.  PADDING is the amount
// of padding above and to the left of the image.  The global ids 
// should be offset by the amount of the padding.  Padding and all
// other dimensions should be the same in both input and output images

#define FILTER_OFFSET ((FILTER_LENGTH-1) >> 1)
#define HALF_WG_H (WG_H >> 1)

__kernel
__attribute__((reqd_work_group_size(WG_W, WG_H, 1)))
void filterY(__global const float* input,
             __global float* output,
             __constant float* filter,
             unsigned int height, unsigned int stride)
{
    const int2 g = (int2) (get_global_id(0), get_global_id(1));
    const int2 l = (int2) (get_local_id(0), get_local_id(1));

    const int pos = g.y*stride + g.x;

    __local float cache[2*WG_H][WG_W];

    // Load a rectangle two workgroups wide
    cache[l.y][l.x] = input[pos - HALF_WG_H*stride];
    cache[l.y+WG_H][l.x] = input[pos + HALF_WG_H*stride];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Wrap the ends, if needed.  Only one level of wrapping
    // is allowed, and the image must be at least HALF_WG_W
    // wide.
    if (g.y < (HALF_WG_H + PADDING))
        cache[l.y][l.x] = cache[WG_H - l.y - 1][l.x];

    const int upperEdge = height + PADDING - 1;

    
    // Check whether we need to mirror the upper half at the 
    // upper end
    if ((g.y + HALF_WG_H) > upperEdge) {

        int overshoot = g.y + HALF_WG_H - upperEdge;

        cache[l.y+WG_H][l.x] 
            = cache[l.y+WG_H - ((overshoot << 1) - 1)][l.x];

        // Check whether the lower half needs the same
        if ((g.y - HALF_WG_H) > upperEdge) {

            overshoot -= WG_H;
            cache[l.y][l.x] = cache[l.y - ((overshoot << 1) - 1)][l.x];

        }

    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Calculate the convolution
    float v = 0.f;
    for (int n = 0; n < FILTER_LENGTH; ++n) 
         v = mad(cache[l.y + n + HALF_WG_H - FILTER_OFFSET][l.x], 
                 filter[FILTER_LENGTH-n-1], v);        

    // Write it to the output
    output[pos] = v;

}
