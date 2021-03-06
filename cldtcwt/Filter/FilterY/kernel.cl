// Copyright (C) 2013 Timothy Gale
// Working group width and height should be defined as WG_W and WG_H;
// the length of the filter is FILTER_LENGTH.  
#define FILTER_OFFSET ((FILTER_LENGTH-1) >> 1)
#define HALF_WG_H (WG_H >> 1)

__kernel
__attribute__((reqd_work_group_size(WG_W, WG_H, 1)))
void filterY(__global const float* input,
             unsigned int inputStart,
             unsigned int stride,
             __global float* output,
             unsigned int outputStart,
             __constant float* filter)
{
    const int2 g = (int2) (get_global_id(0), get_global_id(1));
    const int2 l = (int2) (get_local_id(0), get_local_id(1));

    const int pos = g.y*stride + g.x;
    const int inPos = pos + inputStart;

    __local float cache[2*WG_H][WG_W];

    // Load a rectangle two workgroups wide
    cache[l.y][l.x] = input[inPos - HALF_WG_H*stride];
    cache[l.y+WG_H][l.x] = input[inPos + HALF_WG_H*stride];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Calculate the convolution
    float v = 0.f;
    for (int n = 0; n < FILTER_LENGTH; ++n) 
         v = mad(cache[l.y + n + HALF_WG_H - FILTER_OFFSET][l.x], 
                 filter[FILTER_LENGTH-n-1], v);        

    // Write it to the output
    output[pos + outputStart] = v;

}

