
__kernel void rowDecimateFilter(__read_only image2d_t input,
                                sampler_t inputSampler,
                                __global const float* filter,
                                const int filterLength,
                                __write_only image2d_t output,
                                int offset)
{
    // Coordinates in output frame (rows really are rows, and the column
    // is the number of _pairs_ of numbers along (since the tree outputs are
    // interleaved).
    int x = get_global_id(0);                   
    int y = get_global_id(1);

    // Results for each of the two trees
    float4 out1 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
    float4 out2 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);

    // Apply the filter forward (for the first tree) and backwards (for the
    // second).
    int startX = 4 * x - (filterLength-2) + offset;
    for (int i = 0; i < filterLength; ++i) {
        out1 += filter[filterLength-1-i] *
                read_imagef(input, inputSampler,
                            (int2) (startX+2*i, y));
        out2 += filter[i] *
                read_imagef(input, inputSampler,
                            (int2) (startX+2*i+1, y));
    }

    // Output position is r rows down, plus 2*c along (because the outputs
    // from two trees are interleaved)
    write_imagef(output, (int2) (2*(2*x),   y), out1);
    write_imagef(output, (int2) (2*(2*x+1), y), out2);
}


__kernel void colDecimateFilter(__read_only image2d_t input,
                                sampler_t inputSampler,
                                __global const float* filter,
                                const int filterLength,
                                __write_only image2d_t output,
                                int offset)
{
    // Coordinates in output frame (rows really are rows, and the column
    // is the number of _pairs_ of numbers along (since the tree outputs are
    // interleaved).
    int x = get_global_id(0);                   
    int y = get_global_id(1);

    // Results for each of the two trees
    float4 out1 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
    float4 out2 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);

    // Apply the filter forward (for the first tree) and backwards (for the
    // second).
    int startY = 4 * y - (filterLength-2) + offset;
    for (int i = 0; i < filterLength; ++i) {
        out1 += filter[filterLength-1-i] *
                read_imagef(input, inputSampler,
                            (int2) (x, startY+2*i));
        out2 += filter[i] *
                read_imagef(input, inputSampler,
                            (int2) (x, startY+2*i+1));
    }

    // Output position is r rows down, plus 2*c along (because the outputs
    // from two trees are interleaved)
    write_imagef(output, (int2) (2*x, 2*y), out1);
    write_imagef(output, (int2) (2*x, 2*y+1), out2);
    
       
}

__kernel void rowFilter(__read_only image2d_t input,
                        sampler_t inputSampler,
                        __global const float* filter,
                        const int filterLength,
                        __write_only image2d_t output)
{
    // Row wise filter.  filter must be odd-lengthed
    // Coordinates in output frame
    int x = get_global_id(0);                   
    int y = get_global_id(1);
    
     // Results for each of the two trees
    float4 out = (float4) (0.0f, 0.0f, 0.0f, 0.0f);

    // Apply the filter forward
    int startX = x - (filterLength-1) / 2;
    for (int i = 0; i < filterLength; ++i)
        out += filter[filterLength-1-i] *
                read_imagef(input, inputSampler,
                            (int2) (startX + i, y));

    // Output position is r rows down, plus 2*c along (because the outputs
    // from two trees are interleaved)
    write_imagef(output, (int2) (2*x, y), out);
}


__kernel void colFilter(__read_only image2d_t input,
                        sampler_t inputSampler,
                        __global const float* filter,
                        const int filterLength,
                        __write_only image2d_t output)
{
    // Row wise filter.  filter must be odd-lengthed
    // Coordinates in output frame
    int x = get_global_id(0);                   
    int y = get_global_id(1);
    
     // Results for each of the two trees
    float4 out = (float4) (0.0f, 0.0f, 0.0f, 0.0f);

    // Apply the filter forward
    int startY = y - (filterLength-1) / 2;
    for (int i = 0; i < filterLength; ++i)
        out += filter[filterLength-1-i] *
                read_imagef(input, inputSampler,
                            (int2) (x, startY + i));

    // Output position is r rows down, plus 2*c along (because the outputs
    // from two trees are interleaved)
    write_imagef(output, (int2) (2*x, y), out);
}


__kernel void quadToComplex(__read_only image2d_t input,
                          sampler_t inputSampler,
                          __write_only image2d_t out1Re,
                          __write_only image2d_t out1Im,
                          __write_only image2d_t out2Re,
                          __write_only image2d_t out2Im)
{
    int x = get_global_id(0);                   
    int y = get_global_id(1);

    const float factor = 1.0f / sqrt(2.0f);

    // Sample upper left, upper right, etc
    float4 ul = read_imagef(input, inputSampler, (int2) (  2*x,   2*y));
    float4 ur = read_imagef(input, inputSampler, (int2) (2*x+1,   2*y));
    float4 ll = read_imagef(input, inputSampler, (int2) (  2*x, 2*y+1));
    float4 lr = read_imagef(input, inputSampler, (int2) (2*x+1, 2*y+1));

    write_imagef(out1Re, (int2) (2*x,y), factor * (ul - lr));
    write_imagef(out1Im, (int2) (2*x,y), factor * (ur + ll));
    write_imagef(out2Re, (int2) (2*x,y), factor * (ul + lr));
    write_imagef(out2Im, (int2) (2*x,y), factor * (ur - ll));
}


