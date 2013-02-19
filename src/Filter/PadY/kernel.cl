// PADDING is the amount of padding above and to the left of the image.  The 
// global ids should be offset by the amount of the padding.  


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
__attribute__((reqd_work_group_size(PADDING, PADDING, 1)))
void padY(__global float* image,
          unsigned int height, 
          unsigned int stride)
{
    const int2 g = (int2) (get_global_id(0), get_global_id(1));
    const int2 l = (int2) (get_local_id(0), get_local_id(1));

    __local float cache[PADDING][PADDING];

    if (get_group_id(1) == 0) {

        // Padding to top

        // Read in the square that will contain everything we could want for
        // wrapping
        cache[l.y][l.x] = image[g.y*stride + g.x];

        barrier(CLK_LOCAL_MEM_FENCE);

        // Write it to the output
        image[l.y*stride + g.x] = cache[wrap(PADDING-1-l.y, height)][l.x];

    } else {

        // Padding to bottom

        // Read in the square that will contain everything we could want for
        // wrapping
        cache[l.y][l.x] = image[(PADDING + height - PADDING + l.y) * stride + g.x];

        barrier(CLK_LOCAL_MEM_FENCE);

        // Write it to the output
        image[(PADDING + height + l.y) * stride + g.x] 
            = cache[wrap(height + l.y, height) - (height - PADDING)][l.x];
    }

}

