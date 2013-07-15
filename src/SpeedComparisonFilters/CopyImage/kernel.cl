// Copyright (C) 2013 Timothy Gale
// Working group width and height should be defined as WG_W and WG_H;
// the length of the filter is FILTER_LENGTH.  PADDING is the amount
// of padding above and to the left of the image.  The global ids 
// should be offset by the amount of the padding.  Padding and all
// other dimensions should be the same in both input and output images

#define FILTER_OFFSET ((FILTER_LENGTH-1) >> 1)
#define HALF_WG_W (WG_W >> 1)

__kernel
__attribute__((reqd_work_group_size(WG_W, WG_H, 1)))
void copyImage(__global const float* input,
             __global float* output,
             unsigned int width, unsigned int stride)
{
    const int2 g = (int2) (get_global_id(0), get_global_id(1));
    const int2 l = (int2) (get_local_id(0), get_local_id(1));

    const int pos = g.y*stride + g.x;

    __local float cache[WG_H][WG_W];

    // Load a rectangle one workgroup wide
    cache[l.y][l.x] = input[pos];

    //barrier(CLK_LOCAL_MEM_FENCE);

    // Write it to the output
    output[pos] = cache[l.y][l.x];

}

