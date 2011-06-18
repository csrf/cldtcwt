kernel void square(global float* input,        
                   global float* output,
                   const int width)       
{                                              
    int x = get_global_id(0);                   
    int y = get_global_id(1);

    const int filterSize = 2;

    int indOut = y*(width-filterSize+1) + x;

    output[y*width+x] = input[y*width+x] * input[y*width+x] + x*y/(128.0*128.0);
    //output[y*width+x] = 0.0f;
    //for (int i = 0; i < filterSize; ++i) {
    //    for (int j = 0; j < filterSize; ++j) {
    //        output[indOut] += input[(i+y)*width+x+j];
    //    }
    //}

}                                              


__kernel void rowDecimateFilter(__global const float* input,
                                const int inputStride,
                                __global const float* filter,
                                const int filterLength,
                                __global float* output,
                                const int outputStride)
{
    // Coordinates in output frame (rows really are rows, and the column
    // is the number of _pairs_ of numbers along (since the tree outputs are
    // interleaved).
    int r = get_global_id(0);                   
    int c = get_global_id(1);

    // Results for each of the two trees
    float out1 = 0.0f;
    float out2 = 0.0f;

    // Apply the filter forward (for the first tree) and backwards (for the
    // second).
    int inputPos = r * inputStride + 4 * c;
    for (int i = 0; i < filterLength; ++i) {
        out1 += filter[filterLength-i] * input[inputPos + 2*i];
        out2 += filter[i] * input[inputPos + 1 + 2*i];
    }

    // Output position is r rows down, plus 2*c along (because the outputs
    // from two trees are interleaved.
    int outPos1 = (r * outputStride) + 2 * c;

    output[outPos1]   = out1;
    output[outPos1+1] = out2;

}

__kernel void colDecimateFilter(__global const float* input,
                                const int inputStride,
                                __global const float* filter,
                                const int filterLength,
                                __global float* output,
                                const int outputStride)
{
    // Coordinates in output frame (rows really are rows, and the column
    // is the number of _pairs_ of numbers along (since the tree outputs are
    // interleaved).
    int r = get_global_id(0);                   
    int c = get_global_id(1);

    // Results for each of the two trees
    float out1 = 0.0f;
    float out2 = 0.0f;

    // Apply the filter forward (for the first tree) and backwards (for the
    // second).
    int inputPos = 4 * r * inputStride + c;
    for (int i = 0; i < filterLength; ++i) {
        out1 += filter[filterLength-i] * input[inputPos + 2*i*inputStride];
        out2 += filter[i] * input[inputPos + (1 + 2*i) * inputStride];
    }

    // Output position is r rows down, plus 2*c along (because the outputs
    // from two trees are interleaved.
    int outPos1 = (2 * r * outputStride) + c;

    output[outPos1]   = out1;
    output[outPos1+outputStride] = out2;

}

