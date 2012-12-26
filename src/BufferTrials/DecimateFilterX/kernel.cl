// Working group width and height should be defined as WG_W and WG_H;
// the length of the filter is FILTER_LENGTH.  PADDING is the amount
// of padding above and to the left of the image.  The global ids 
// should be offset by the amount of the padding.  Padding and all
// other dimensions should be the same in both input and output images

#define FILTER_OFFSET (FILTER_LENGTH-2)
#define HALF_WG_W (WG_W >> 1)


int wrap(int n, int width)
{
    // Perform symmetric extension of an index, if needed.  The input n
    // must not be negative.
    if (n < width)
        return n;
    else {
        int tmp = n % (2*width);
        return min(tmp, 2*width - 1 - tmp);
    }
}


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

    // Extract evens in order
    const int evenAddr = l.x >> 1;

    // Extract odds backwards, and with pairs in reverse order
    // (so as to avoid bank conflicts when reading later)
    const int oddAddr = (4*WG_W - 1 - evenAddr) ^ 1;

    const int d = (l.x & 1) ? -1 : 1; // Direction to move the block in
    const int p = (l.x & 1) ? oddAddr : evenAddr;

    cache[l.y][p                 ] = input[pos - WG_W];
    cache[l.y][p + d*  (WG_W / 2)] = input[pos];
    cache[l.y][p + d*2*(WG_W / 2)] = input[pos + WG_W];
    cache[l.y][p + d*3*(WG_W / 2)] = input[pos + 2*WG_W];

    barrier(CLK_LOCAL_MEM_FENCE);

    float v = 0.f;

    int baseOffset = l.x + (HALF_WG_W  - (FILTER_LENGTH >> 1) + 1)
                         + (l.x & 1) * (3*WG_W - 3);

    const int offset1 = select(baseOffset ^ 1, baseOffset, l.x & 1);

    baseOffset += 1;
    const int offset2 = select(baseOffset ^ 1, baseOffset, l.x & 1);

    for (int n = 0; n < FILTER_LENGTH; n += 2) 
        v += filter[n] * cache[l.y][offset1+n];
        
    for (int n = 0; n < FILTER_LENGTH; n += 2) 
        v += filter[n+1] * cache[l.y][offset2+n];

    // Write it to the output
    output[g.y*outStride + g.x] = v;

}

