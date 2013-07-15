// Copyright (C) 2013 Timothy Gale
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
    // again.  These are split into even (based from 0, so we start with 
    // even) and odd x-coords, so that the first two blocks of the output
    // are the even columns.  The second two blocks are the odd columns.
    // The second two blocks have been reversed along the x-axis, and adjacent
    // columns within that are swapped.
    //
    // l is the coordinate within the block.

    // Calculate output x coordinates whether reading from an even or odd
    // address
    const int evenAddr = l.x >> 1;

    // Extract odds backwards, and with pairs in reverse order
    // (so as to avoid bank conflicts when reading later)
    const int oddAddr = (4*WG_W - 1 - evenAddr) ^ 1;

    // Direction to move the block in: -1 for odds, 1 for evens
    const int d = select(WG_W / 2, -WG_W / 2, l.x & 1); 
    const int p = select(evenAddr,   oddAddr, l.x & 1);

    cache[l.y][p    ] = *(readPos-WG_W);
    cache[l.y][p+  d] = *(readPos);
    cache[l.y][p+2*d] = *(readPos+WG_W);
    cache[l.y][p+3*d] = *(readPos+2*WG_W);
}


inline int2 filteringStartPositions(int x)
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

    // Calculate positions to read coefficients from
    int baseOffset = x + ((WG_W / 2) - (FILTER_LENGTH / 2) + 1)
                         + (x & 1) * (3*WG_W - 1 - 2*x);

    // Starting locations for first and second trees
    return (int2) (select(baseOffset, baseOffset ^ 1, x & 1),
                   select(baseOffset + 1, (baseOffset + 1) ^ 1, x & 1));
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
    int2 offset = filteringStartPositions(l.x);

    // Convolve 
    float v = 0.f;

    // Even filter locations first...
    for (int n = 0; n < FILTER_LENGTH; n += 2) 
        v += filter[n] * cache[l.y][offset.s0+n];
        
    // ...then odd
    for (int n = 0; n < FILTER_LENGTH; n += 2) 
        v += filter[n+1] * cache[l.y][offset.s1+n];

    // Write it to the output
    output[g.y*outputStride + (g.x ^ SWAP_TREE_1) + outputStart] = v;

}

