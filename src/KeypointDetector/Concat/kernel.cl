// Copyright (C) 2013 Timothy Gale
__kernel
void concat(__read_only global float* inputArray,
            __write_only global float* outputArray,
            __read_only global unsigned int* cumCounts,
            unsigned int cumCountsIndex,
            unsigned int numFloatsPerItem)
{
    // cumCounts[cumCountsIndex] contains the item number to start outputting
    // to; cumCounts[cumCountsIndex+1] one beyond the end of number of items.
    // Copies from inputArray to this outputArray.  numFloats is the number
    // of floats per item

    size_t c0 = cumCounts[cumCountsIndex],
           c1 = cumCounts[cumCountsIndex+1];

    size_t startOutputPos = numFloatsPerItem * c0;
    size_t numFloatsToCopy = numFloatsPerItem * (c1 - c0);

    for (int n = get_global_id(0); 
             n < numFloatsToCopy; 
             n += get_global_size(0)) 
        outputArray[startOutputPos+n] = inputArray[n];

}





