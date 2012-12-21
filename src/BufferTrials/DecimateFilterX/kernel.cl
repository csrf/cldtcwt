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

    // Load a rectangle two workgroups wide
    cache[l.y][l.x]        = input[pos - WG_W];
    cache[l.y][l.x+WG_W]   = input[pos];
    cache[l.y][l.x+2*WG_W] = input[pos + WG_W];
    cache[l.y][l.x+3*WG_W] = input[pos + 2*WG_W];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Wrap the bottom end, if needed.      
    int offset = g.x - (WG_W + PADDING);
    if (offset < 0) {
        offset = -1 - offset;
        cache[l.y][l.x] = cache[l.y][WG_W + wrap(offset, width)];
    }

    // Position within the cache of the start of invalid data
    const int threshold =  width - 2 * WG_W * get_group_id(0) + WG_W;

    if ((l.x + 3*WG_W) >= threshold) {
        
        int readpos = wrap(g.x - PADDING + 3*WG_W, width);

        int localReadpos = width-readpos+threshold;

        // Make sure we don't read an invalid location!
        cache[l.y][l.x+3*WG_W] = cache[l.y][max(localReadpos, 0)];

        if ((l.x + 2*WG_W) >= threshold) {

            int readpos = wrap(g.x - PADDING + 2*WG_W, width);

            int localReadpos = width-readpos+threshold;

            // Make sure we don't read an invalid location!
            cache[l.y][l.x+2*WG_W] = cache[l.y][max(localReadpos, 0)];


            if ((l.x + WG_W) >= threshold) {

                int readpos = wrap(g.x - PADDING + WG_W, width);

                int localReadpos = width-readpos+threshold;

                // Make sure we don't read an invalid location!
                cache[l.y][l.x+WG_W] = cache[l.y][max(localReadpos, 0)];

            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Calculate the convolution
    float v = 0.f;
    for (int n = 0; n < (2*FILTER_LENGTH); n += 2) 
        v = mad(cache[l.y][l.x + n + WG_W - FILTER_OFFSET], 
                select(filter[n+1], filter[n], l.x & 1), v);      
    

    // Write it to the output
    output[g.y*outStride + g.x] = v;

}

