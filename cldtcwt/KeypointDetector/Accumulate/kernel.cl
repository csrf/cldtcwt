// Copyright (C) 2013 Timothy Gale
__kernel
void accumulate(__global __read_only unsigned int* inputs,
                unsigned int numInputs,
                __global __write_only unsigned int* cumSum,
                unsigned int maxSum)
{
    // Write out the cumulative sum of inputs, with the first output zero,
    // never exceeding maxSum.

    unsigned int sum = 0;

    for (int n = 0; n < numInputs; ++n) {
        
        // Record the sum so far
        cumSum[n] = sum;
        sum += inputs[n];

        // Make sure the output never exceeds the maximum output
        sum = min(sum, maxSum);
    }

    // Record the last position
    cumSum[numInputs] = sum;
}

