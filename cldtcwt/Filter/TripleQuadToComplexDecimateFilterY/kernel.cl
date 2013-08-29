// Copyright (C) 2013 Timothy Gale
// Working group width and height should be defined as WG_W and WG_H;
// the length of the filter is FILTER_LENGTH.  

// Choosing to swap the outputs of the two trees is selected by defining
// SWAP_TREE_1

#ifndef SWAP_TREE_0 
    #define SWAP_TREE_0 0
#else
    #define SWAP_TREE_0 1
#endif

#ifndef SWAP_TREE_1 
    #define SWAP_TREE_1 0
#else
    #define SWAP_TREE_1 1
#endif

#ifndef SWAP_TREE_2 
    #define SWAP_TREE_2 0
#else
    #define SWAP_TREE_2 1
#endif


// Used to swap the right tree over according to z-idx
__constant bool swapTree[] = {SWAP_TREE_0, SWAP_TREE_1, SWAP_TREE_2};


void loadFourBlocks(__global const float* readPos, size_t stride, int2 l,
                    __local float cache[4*WG_H][WG_W])
{
    // Load four blocks of WG_W x WG_H into cache: the first from the
    // locations specified one workgroup height above readPos; the 
    // second from readPos; the third below, and fourth below
    // again.  These are split into even (based from 0, so we start with 
    // even) and odd y-coords, so that the first two blocks of the output
    // are the even rows.  The second two blocks are the odd rows.
    // The second two blocks have been reversed along the y-axis, and adjacent
    // rows within that are swapped.
    //
    // l is the coordinate within the block.

    // Calculate output y coordinates whether reading from an even or odd
    // address
    const int evenAddr = l.y >> 1;

    // Extract odds backwards, and with pairs in reverse order
    // (so as to avoid bank conflicts when reading later)
    const int oddAddr = (4*WG_W - 1 - evenAddr) ^ 1;

    // Direction to move the block in: -1 for odds, 1 for evens
    const int d = select(WG_W / 2, -WG_W / 2, l.y & 1); 
    const int p = select(evenAddr,   oddAddr, l.y & 1);

    cache[p      ][l.x] = *(readPos-WG_H*stride);
    cache[p +   d][l.x] = *(readPos);
    cache[p + 2*d][l.x] = *(readPos+WG_H*stride);
    cache[p + 3*d][l.x] = *(readPos+2*WG_H*stride);
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
void decimateFilterY(__global const float* input,
                     unsigned int inputStart,
                     unsigned int inputPitch,
                     unsigned int inputStride,
                     __global float* output,
                     unsigned int outputStart,
                     unsigned int outputPitch,
                     unsigned int outputStride,
                     unsigned int outputWidth,
                     unsigned int outputHeight,
                     __constant float* filter)
{
    const int2 g = (int2) (get_global_id(0), get_global_id(1));
    const int2 l = (int2) (get_local_id(0), get_local_id(1));

    __local float cache[4*WG_H][WG_W];

    // Decimation means we also need to move along according to
    // workgroup number (since we move along the input faster than
    // along the output matrix).
    const int pos = (g.y + get_group_id(1) * WG_H) * inputStride + g.x
                    + inputStart + inputPitch * get_group_id(2);

    // Read into local memory
    loadFourBlocks(&input[pos], inputStride, l, cache);

    barrier(CLK_LOCAL_MEM_FENCE);

    // Work out where we need to start the convolution from
    int2 offset = filteringStartPositions(l.y);

    // Convolve 
    float v = 0.f;

    // filter contains the filters for all inputs; we want to use
    // the z-index along
    //filter += get_group_id(2) * FILTER_LENGTH;
    if (get_group_id(2) == 0) {

        // Even filter locations first...
        for (int n = 0; n < FILTER_LENGTH; n += 2) 
            v += filter[n] * cache[offset.s0+n][l.x];
            
        // ...then odd
        for (int n = 0; n < FILTER_LENGTH; n += 2) 
            v += filter[n+1] * cache[offset.s1+n][l.x];

    } else if (get_group_id(2) == 1) {

        // Even filter locations first...
        for (int n = 0; n < FILTER_LENGTH; n += 2) 
            v += filter[FILTER_LENGTH + n] * cache[offset.s0+n][l.x];
            
        // ...then odd
        for (int n = 0; n < FILTER_LENGTH; n += 2) 
            v += filter[FILTER_LENGTH + n+1] * cache[offset.s1+n][l.x];

    } else {

        // Even filter locations first...
        for (int n = 0; n < FILTER_LENGTH; n += 2) 
            v += filter[2*FILTER_LENGTH + n] * cache[offset.s0+n][l.x];
            
        // ...then odd
        for (int n = 0; n < FILTER_LENGTH; n += 2) 
            v += filter[2*FILTER_LENGTH + n+1] * cache[offset.s1+n][l.x];

    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Now we want to share the results
    cache[l.y ^ swapTree[get_group_id(2)]][l.x] = v;

    barrier(CLK_LOCAL_MEM_FENCE);

    int2 outPos = g >> 1;

    // Output only using the top right of each square of four pixels,
    // and only within the confines of the image
    if ((outPos.x < outputWidth) & (outPos.y < outputHeight)) {

        // Sample upper left, upper right, etc
        // More comprehensible version:
        /*float ul = cache[l.y][l.x];
        float ur = cache[l.y][l.x+1];
        float ll = cache[l.y+1][l.x];
        float lr = cache[l.y+1][l.x+1];

        // Combine into complex pairs
        output0[outPos.x * 2 + outPos.y * outputWidth * 2]
            = factor * (ul - lr);
        output0[outPos.x * 2 + outPos.y * outputWidth * 2 + 1]
            = factor * (ur + ll);
            
        output1[outPos.x * 2 + outPos.y * outputWidth * 2]
            = factor * (ul + lr);
        output1[outPos.x * 2 + outPos.y * outputWidth * 2 + 1]
            = factor * (ur - ll);*/

        const float factor = 1.0f / sqrt(2.0f);

        // Version which avoids branches:
        int y = l.y & ~1;

        // Load upper value (u?) into a, lower (l?) into b
        float a = cache[y][l.x];
        float b = cache[y ^ 1][l.x ^ 1];

        float rplus  = a + b;
        float rminus = a - b;

        // Select the right subband for output.  The second output is in the 
        // opposite subband
        unsigned int start = 
            outputStart + outputPitch 
                    * select(get_group_id(2), (size_t)(5lu - get_group_id(2)), (size_t)(l.y & 1ul));
        
        // Add or subtract, and place in appropriate output
        output[2 * (start + outPos.x + outPos.y*outputStride) 
               + (l.x & 1)]
            = factor * (((l.x & 1) ^ (l.y & 1))? rplus : rminus);

    }

}

