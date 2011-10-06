
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


__kernel void cornernessMap(__read_only image2d_t sb1re,
                            __read_only image2d_t sb1im,
                            __read_only image2d_t sb2re,
                            __read_only image2d_t sb2im,
                            __read_only image2d_t sb3re,
                            __read_only image2d_t sb3im,
                            __read_only image2d_t sb4re,
                            __read_only image2d_t sb4im,
                            __read_only image2d_t sb5re,
                            __read_only image2d_t sb5im,
                            __read_only image2d_t sb6re,
                            __read_only image2d_t sb6im,
                            sampler_t inputSampler,
                            __write_only image2d_t cornerMap)
{
    int2 coord = (int2) (get_global_id(0), get_global_id(1));

    float4 f1r = read_imagef(sb1re, inputSampler, coord);
    float4 f1i = read_imagef(sb1im, inputSampler, coord);
    float4 f2r = read_imagef(sb2re, inputSampler, coord);
    float4 f2i = read_imagef(sb2im, inputSampler, coord);
    float4 f3r = read_imagef(sb3re, inputSampler, coord);
    float4 f3i = read_imagef(sb3im, inputSampler, coord);
    float4 f4r = read_imagef(sb4re, inputSampler, coord);
    float4 f4i = read_imagef(sb4im, inputSampler, coord);
    float4 f5r = read_imagef(sb5re, inputSampler, coord);
    float4 f5i = read_imagef(sb5im, inputSampler, coord);
    float4 f6r = read_imagef(sb6re, inputSampler, coord);
    float4 f6i = read_imagef(sb6im, inputSampler, coord);

    float4 result = fmin(f1r.x * f1r.x + f1i.x * f1i.x,
                    fmin(f2r.x * f2r.x + f2i.x * f2i.x,
                    fmin(f3r.x * f3r.x + f3i.x * f3i.x,
                    fmin(f4r.x * f4r.x + f4i.x * f4i.x,
                    fmin(f5r.x * f5r.x + f5i.x * f5i.x,
                         f6r.x * f6r.x + f6i.x * f6i.x)))));
    
    write_imagef(cornerMap, (int2) (coord.x * 2, coord.y), sqrt(result));
}








