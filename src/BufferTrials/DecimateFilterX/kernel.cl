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



void loadFourBlocks(int pos,  __global float* input,
                    __local float cache[WG_H][4*WG_W],
                    int2 l, int pad, bool twiddleTree2)
{
    // Load four blocks of WG_H x WG_H into cache, with this work item
    // working around pos.

    // l contains the x and y coordinates of this workitem within the group
    // cache is the region to load into.  Even x coordinates are loaded into

    // Extract evens in order (left half of cache), and odds in reverse order
    // (right half of cache).  If extending (pad == 1) we need to swap the trees
    // to put everything in the right place.

    // If twiddling Tree 2 (to improve __local bank access efficiency by avoiding
    // conflicts), swap each pair of values stored for Tree 2.

    const int evenAddr = l.x >> 1;

    // Extract odds backwards, and with pairs in reverse order
    // (so as to avoid bank conflicts when reading later)
    const int oddAddr = (4*WG_W - 1 - evenAddr) ^ twiddleTree2;

    // We want to store into the reverse order if on an odd address;
    // but the trees are swapped over if we have to pad
    bool storeBackwards = (l.x & 1) ^ pad;

    const int d = 1 - 2*storeBackwards; 

    // Direction to move the block in: -1 for odds, 1 for evens
    const int p = select(evenAddr, oddAddr, storeBackwards);

    cache[l.y][p                 ] = input[pos - WG_W];
    cache[l.y][p + d*  (WG_W / 2)] = input[pos];
    cache[l.y][p + d*2*(WG_W / 2)] = input[pos + WG_W];
    cache[l.y][p + d*3*(WG_W / 2)] = input[pos + 2*WG_W];
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
    int baseOffset = x + ((WG_W / 2) - (FILTER_LENGTH / 2) + 1)
                         - pad
                         + (x & 1) * (3*WG_W - 1 - 2*x + pad);

    // Starting locations for first and second trees
    return (int2) (select(baseOffset, baseOffset ^ twiddleTree2, x & 1),
                   select(baseOffset + 1, (baseOffset + 1) ^ twiddleTree2, x & 1));
}



__kernel
__attribute__((reqd_work_group_size(WG_W, WG_H, 1)))
void decimateFilterX(__global const float* input,
             __global float* output,
             __constant float* filter,
             unsigned int width, 
             unsigned int stride,
             unsigned int outStride,
             int pad)
{
    // pad should be 0 or 1: 1 if the algorithm should pretend the 
    // image extends one of width on both sides.

    const int2 g = (int2) (get_global_id(0), get_global_id(1));
    const int2 l = (int2) (get_local_id(0), get_local_id(1));

    // Decimation means we also need to move along according to
    // workgroup number (since we move along the input faster than
    // along the output matrix).
    const int pos = g.y*stride + g.x + get_group_id(0) * WG_W;

    // Usually we want to swap the pairs of values in the second, reversed,
    // tree so as to keep one tree accessing odds while the other accesses
    // events.  However, if padding, the first tree starts accessing one 
    // lower down.  This means we don't want to do the swapping then.
    const bool twiddleTree2 = pad == 0;

    __local float cache[WG_H][4*WG_W];

    loadFourBlocks(pos, input, cache, l, pad, twiddleTree2);

    barrier(CLK_LOCAL_MEM_FENCE);

    // Work out where we need to start the convolution from
    int2 offset = filteringStartPositions(l.x, pad, twiddleTree2);

    // If we want to swap the trees over, the easiest way is to 
    // swap the LSB of l.x, and recalculate
    int2 offsetSwapOutputs = filteringStartPositions(l.x ^ 1, pad, twiddleTree2);

    // Convolve 

    float v = 0.f;

    // Even filter locations first...
    for (int n = 0; n < FILTER_LENGTH; n += 2) 
        v += filter[n] * cache[l.y][TREE_1_OFFSET.s0+n];
        
    // ...then odd
    for (int n = 0; n < FILTER_LENGTH; n += 2) 
        v += filter[n+1] * cache[l.y][TREE_1_OFFSET.s1+n];

    // Write it to the output
    output[g.y*outStride + g.x] = v;

}

