// Working group width and height should be defined as WG_W and WG_H;
// the length of the filter is FILTER_LENGTH.  PADDING is the amount
// of padding above and to the left of the image.  The global ids 
// should be offset by the amount of the padding.  Padding and all
// other dimensions should be the same in both input and output images

// Choosing to swap the outputs of the two trees is selected by defining
// SWAP_TREE_1

#ifndef SWAP_TREE_1 
    #define TREE_1_OFFSET offset
#else
    #define TREE_1_OFFSET offsetSwapOutputs
#endif



void loadFourBlocks(__global const float* readPos, size_t stride,
                    __local float cache[4*WG_H][WG_W],
                    int2 l, int pad, bool twiddleTree2)
{
    // Load four blocks of WG_H x WG_H into cache, with this work item
    // working around readPos.

    // l contains the x and y coordinates of this workitem within the group
    // cache is the region to load into.  Even x coordinates are loaded into

    // Extract evens in order (left half of cache), and odds in reverse order
    // (right half of cache).  If extending (pad == 1) we need to swap the trees
    // to put everything in the right place.

    // If twiddling Tree 2 (to improve __local bank access efficiency by avoiding
    // conflicts), swap each pair of values stored for Tree 2.

    const int evenAddr = l.y >> 1;

    // Extract odds backwards, and with pairs in reverse order
    // (so as to avoid bank conflicts when reading later)
    const int oddAddr = (4*WG_H - 1 - evenAddr) ^ twiddleTree2;

    // We want to store into the reverse order if on an odd address;
    // but the trees are swapped over if we have to pad
    int storeBackwards = (l.y & 1) ^ pad;

    const int d = 1 - 2*storeBackwards; 

    // Direction to move the block in: -1 for odds, 1 for evens
    const int p = select(evenAddr, oddAddr, storeBackwards);

    cache[p                 ][l.x] = *(readPos-WG_H*stride);
    cache[p + d*  (WG_H / 2)][l.x] = *(readPos);
    cache[p + d*2*(WG_H / 2)][l.x] = *(readPos+WG_H*stride);
    cache[p + d*3*(WG_H / 2)][l.x] = *(readPos+2*WG_H*stride);
}


inline int2 filteringStartPositions(int x, int pad, bool twiddleTree2)
{
    // Padding is whether we are using symmetric extension (0 or 1)
    // Twiddle Tree 2 is whether or not the second tree has had its pairs
    // swapped (for an optimisation to avoid shared memory bank conflicts).
    // x is the x position within the workgroup

    // Each position along x should be calculating the output for that position.
    // The two trees are stored in the left and right halves of a 4 WG_(dim)
    // with the second tree (right) in reverse.  If padding, the odd coeffients 
    // for Tree 2 need to be read one further to the left.

    // Calculate positions to read coefficients from
    int baseOffset = x + ((WG_H / 2) - (FILTER_LENGTH / 2) + 1)
                         - pad
                         + (x & 1) * (3*WG_H - 1 - 2*x + pad);

    // Starting locations for first and second trees
    return (int2) (select(baseOffset, baseOffset ^ twiddleTree2, x & 1),
                   select(baseOffset + 1, (baseOffset + 1) ^ twiddleTree2, x & 1));
}



__kernel
__attribute__((reqd_work_group_size(WG_W, WG_H, 1)))
void decimateFilterY(__global const float* input,
                     __global float* output0,
                     __global float* output1,
                     unsigned int outputWidth,
                     unsigned int outputHeight,
                     __constant float* filter,
                     unsigned int height, 
                     unsigned int stride,
                     int pad)
{
    // pad should be 0 or 1: 1 if the algorithm should pretend the 
    // image extends one of width on both sides.

    const int2 g = (int2) (get_global_id(0), get_global_id(1));
    const int2 l = (int2) (get_local_id(0), get_local_id(1));

    // Decimation means we also need to move along according to
    // workgroup number (since we move along the input faster than
    // along the output matrix).
    const int pos = (g.y + get_group_id(1) * WG_H) * stride + g.x;

    // Usually we want to swap the pairs of values in the second, reversed,
    // tree so as to keep one tree accessing odds while the other accesses
    // events.  However, if padding, the first tree starts accessing one 
    // lower down.  This means we don't want to do the swapping then.
    const bool twiddleTree2 = pad == 0;

    __local float cache[4*WG_H][WG_W];

    // Read into local memory
    loadFourBlocks(&input[pos], stride, cache, l, pad, twiddleTree2);

    barrier(CLK_LOCAL_MEM_FENCE);

    // Work out where we need to start the convolution from
    int2 offset = filteringStartPositions(l.y, pad, twiddleTree2);

    // If we want to swap the trees over, the easiest way is to 
    // swap the LSB of l.x, and recalculate
    int2 offsetSwapOutputs = filteringStartPositions(l.y ^ 1, pad, twiddleTree2);

    // Convolve 

    float v = 0.f;

    // Even filter locations first...
    for (int n = 0; n < FILTER_LENGTH; n += 2) 
        v += filter[n] * cache[TREE_1_OFFSET.s0+n][l.x];
        
    // ...then odd
    for (int n = 0; n < FILTER_LENGTH; n += 2) 
        v += filter[n+1] * cache[TREE_1_OFFSET.s1+n][l.x];
    
    barrier(CLK_LOCAL_MEM_FENCE);

    // Now we want to share the results
    cache[l.y][l.x] = v;

    barrier(CLK_LOCAL_MEM_FENCE);

    int2 outPos = (g - PADDING) >> 1;

    // Output only using the top right of each square of four pixels,
    // and only within the confines of the image
    if ((outPos.x < (2*outputWidth)) & (outPos.y < outputHeight)) {

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

        // Version which avoids branches:
        int y = l.y & ~1;

        // Load upper value (u?) into a, lower (l?) into b
        float a = cache[y][l.x];
        float b = cache[y ^ 1][l.x ^ 1];
        float sign = ((l.x & 1) ^ (l.y & 1))? 1.f : -1.f;

        const float factor = 1.0f / sqrt(2.0f);

        __global float* output = (l.y & 1)? output1 : output0;
        
        // Add or subtract, and place in appropriate output
        output[outPos.x * 2 + (l.x & 1) + outPos.y * outputWidth * 2]
            = factor * (a + sign * b);

    }

}

