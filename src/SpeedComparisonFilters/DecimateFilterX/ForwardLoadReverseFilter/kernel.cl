// Working group width and height should be defined as WG_W and WG_H;
// the length of the filter is FILTER_LENGTH.  

// Choosing to swap the outputs of the two trees is selected by defining
// SWAP_TREE_1

#ifdef SWAP_TREE_1 
    #define SWAP_TREE_1 1
#else
    #define SWAP_TREE_1 0
#endif



void loadFourBlocks(__global const float* readPos,
                    int2 l,
                    __local float cache[WG_H][4*WG_W])
{
    // Load four blocks of WG_W x WG_H into cache: the first from the
    // locations specified one workgroup width to the left of readPos; the 
    // second from readPos; the third to the right, and fourth to the right
    // again.  

    // l is the coordinate within the block.

    cache[l.y][l.x       ] = *(readPos-WG_W);
    cache[l.y][l.x+  WG_W] = *(readPos);
    cache[l.y][l.x+2*WG_W] = *(readPos+WG_W);
    cache[l.y][l.x+3*WG_W] = *(readPos+2*WG_W);
}


inline int filteringStartPosition(int x)
{
    // Calculates the positions within the block to start filtering from,
    // for the even and odd coefficients in the filter respectively given
    // in s0 and s1.  Assumes the four-workgroup format loaded in
    // loadFourBlocks.
    //
    // x is the x position within the workgroup.

    // Each position along x should be calculating the output for that position.
    // The two trees are stored in the left and right halves of a 4 WG_(dim)
    // with the second tree (right) in reverse.

    // Calculate position to read coefficients from
    return select(2*(x + ((WG_W / 2) - (FILTER_LENGTH / 2) + 1)),
                  4*WG_W - (2*(WG_W / 2 - (FILTER_LENGTH / 2) + WG_W - x) + 1)
                   - (FILTER_LENGTH * 2 - 2),
                  x & 1);
}



__kernel
__attribute__((reqd_work_group_size(WG_W, WG_H, 1)))
void decimateFilterX(__global const float* input,
                     unsigned int inputStart,
                     unsigned int inputStride,
                     __global float* output,
                     unsigned int outputStart,
                     unsigned int outputStride,
                     __constant float* filter)
{
    const int2 g = (int2) (get_global_id(0), get_global_id(1));
    const int2 l = (int2) (get_local_id(0), get_local_id(1));

    __local float cache[WG_H][4*WG_W];

    // Decimation means we also need to move along according to
    // workgroup number (since we move along the input faster than
    // along the output matrix).
    const int pos = g.y*inputStride + g.x + get_group_id(0) * WG_W
                    + inputStart;

    // Read into local memory
    loadFourBlocks(&input[pos], l, cache);

    barrier(CLK_LOCAL_MEM_FENCE);

    // Work out where we need to start the convolution from
    int offset = filteringStartPosition(l.x);

    // Convolve 
    float v = 0.f;

    // Filter forwards if even, backwards if odd
    int s = select(0, FILTER_LENGTH-1, l.x & 1);
    int d = select(1, -1, l.x & 1);

    for (int n = 0; n < FILTER_LENGTH; ++n) 
        v += filter[s + d*n] * cache[l.y][offset+2*n];

    // Write it to the output
    output[g.y*outputStride + (g.x ^ SWAP_TREE_1) + outputStart] = v;

}

