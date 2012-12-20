// Working group width and height should be defined as WG_W and WG_H;
// the length of the filter is FILTER_LENGTH.  PADDING is the amount
// of padding above and to the left of the image.  The global ids 
// should be offset by the amount of the padding.  Padding and all
// other dimensions should be the same in both input and output images

#define FILTER_OFFSET (FILTER_LENGTH-2)
#define HALF_WG_W (WG_W >> 1)

__kernel
__attribute__((reqd_work_group_size(WG_W, WG_H, 1)))
void decimateFilterX(__global const float* input,
             __global float* output,
             __constant float* filter,
             unsigned int width, 
             unsigned int stride,
             unsigned int outStride)
{
    const int2 g = (int2) (get_global_id(0), get_global_id(1));
    const int2 l = (int2) (get_local_id(0), get_local_id(1));

    // Decimation means we also need to move along according to
    // workgroup number (since we move along the input faster than
    // along the output matrix).
    const int pos = g.y*stride + g.x + get_group_id(0) * WG_W;

    __local float cache[WG_H][4*WG_W];

    // Load a rectangle two workgroups wide
    cache[l.y][l.x]        = input[pos - WG_W];
    cache[l.y][l.x+WG_W]   = input[pos];
    cache[l.y][l.x+2*WG_W] = input[pos + WG_W];
    cache[l.y][l.x+3*WG_W] = input[pos + 2*WG_W];

    barrier(CLK_LOCAL_MEM_FENCE);
#if 0
    // Wrap the ends, if needed.  Only one level of wrapping
    // is allowed, and the image must be at least HALF_WG_W
    // wide.
    if (g.x < (HALF_WG_W + PADDING))
        cache[l.y][l.x] = cache[l.y][WG_W - l.x - 1];

    const int upperEdge = width + PADDING - 1;

    
    // Check whether we need to mirror the upper half at the 
    // upper end
    if ((g.x + HALF_WG_W) > upperEdge) {

        int overshoot = g.x + HALF_WG_W - upperEdge;

        cache[l.y][l.x+WG_W] = cache[l.y][l.x+WG_W - 
                                      ((overshoot << 1) - 1)];

        // Check whether the lower half needs the same
        if ((g.x - HALF_WG_W) > upperEdge) {

            overshoot -= WG_W;
            cache[l.y][l.x] = cache[l.y][l.x - ((overshoot << 1) - 1)];

        }

    }

    barrier(CLK_LOCAL_MEM_FENCE);

#endif

    // Calculate the convolution
    float v = 0.f;
    for (int n = 0; n < (2*FILTER_LENGTH); n += 2) 
         v = mad(cache[l.y][l.x + n + WG_W - FILTER_OFFSET], 
                 filter[n], v);        

    // Write it to the output
    output[g.y*outStride + g.x] = v;

}

