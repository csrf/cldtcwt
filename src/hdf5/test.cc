#include <iostream>
#include <vector>
#include <string>
#include "hdfwriter.h"



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

