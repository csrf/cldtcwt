// Copyright (C) 2013 Timothy Gale
#include <iostream>
#include <vector>
#include <string>
#include "hdf5/hdfwriter.h"



int main()
{
    HDFWriter hwtest("test2.h5", 10);

    std::vector<float> zeros(20, 0.f);
    zeros[2] = 1.f;

    hwtest.append(2, &zeros[0], &zeros[0]);
    hwtest.append(2, &zeros[0], &zeros[0]);
    hwtest.append(1, &zeros[0], &zeros[0]);

    return 0;
}

