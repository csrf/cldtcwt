#include "referenceImplementation.h"

#include <cmath>


static unsigned int wrap(int n, int width)
{
    // Wrap so that the pattern goes
    // forwards-backwards-forwards-backwards etc, with the end
    // values repeated.
    
    int result = n % (2 * width);

    // Make sure we get the positive result
    if (result < 0)
        result += 2*width;

    return std::min(result, 2*width - result - 1);
}




Eigen::ArrayXXf convolveRows(const Eigen::ArrayXXf& in, 
                             const std::vector<float>& filter)
{
    size_t offset = (filter.size() - 1) / 2;

    Eigen::ArrayXXf output(in.rows(), in.cols());

    // Pad the input
    Eigen::ArrayXXf padded(in.rows(), in.cols() + filter.size() - 1);

    for (int n = 0; n < padded.cols(); ++n) 
        padded.col(n) = in.col(wrap(n - offset, in.cols()));

    // For each output pixel
    for (size_t r = 0; r < in.rows(); ++r)
        for (size_t c = 0; c < in.cols(); ++c) {

            // Perform the convolution
            float v = 0.f;
            for (size_t n = 0; n < filter.size(); ++n)
                v += filter[filter.size()-n-1]
                        * padded(r, c+n);

            output(r,c) = v;
        }

    return output;
}



Eigen::ArrayXXf convolveCols(const Eigen::ArrayXXf& in, 
                             const std::vector<float>& filter)
{
    return convolveRows(in.transpose(), filter).transpose();
}



Eigen::ArrayXXf decimateConvolveRows
                            (const Eigen::ArrayXXf& in, 
                             const std::vector<float>& filter,
                             bool swapOutputs)
{
    // If extending, we want to create an extra output by
    // taking an extra sample from each end.  Symmetric is
    // whether the reversed filter output should come first in
    // the pairs or second

    bool extend = (in.cols() % 4) != 0;

    size_t offset = filter.size() - 2 + (extend? 1 : 0);

    Eigen::ArrayXXf output(in.rows(), 
            (in.cols() + (extend? 2 : 0)) / 2);

    // Pad the input
    Eigen::ArrayXXf padded(in.rows(), in.cols() + 2 * offset);

    for (int n = 0; n < padded.cols(); ++n) 
        padded.col(n) = in.col(wrap(n - offset, in.cols()));

    // For each pair of output pixels
    for (size_t r = 0; r < output.rows(); ++r)
        for (size_t c = 0; c < output.cols(); c += 2) {

            // Perform the convolution
            float v1 = 0.f, v2 = 0.f;

            for (size_t n = 0; n < filter.size(); ++n) {
                v1 += filter[filter.size()-n-1]
                        * padded(r, 2*c+2*n);
               
                v2 += filter[n]
                        * padded(r, 2*c+2*n+1);
            }

            output(r,c) = swapOutputs? v2 : v1;
            output(r,c+1) = swapOutputs? v1 : v2;
        }

    return output;
}



Eigen::ArrayXXf decimateConvolveCols
                            (const Eigen::ArrayXXf& in, 
                             const std::vector<float>& filter,
                             bool swapOutputs)
{
    return decimateConvolveRows(in.transpose(), filter, swapOutputs)
                .transpose();   
}



std::tuple<Eigen::ArrayXXcf, Eigen::ArrayXXcf>
    quadToComplex(const Eigen::ArrayXXf& in)
{
    // Convert an interleaved set of four trees into two
    // complex subbands.

    Eigen::ArrayXXcf sb0(in.rows() / 2, in.cols() / 2), 
                     sb1(in.rows() / 2, in.cols() / 2);

    for (int r = 0; r < in.rows(); r += 2)
        for (int c = 0; c < in.cols(); c += 2) {

            float ul = in(r,c);
            float ur = in(r,c+1);
            float ll = in(r+1,c);
            float lr = in(r+1,c+1);

            sb0(r/2,c/2) = std::complex<float>(ul - lr, ur + ll) / sqrtf(2.f);
            sb1(r/2,c/2) = std::complex<float>(ul + lr, ur - ll) / sqrtf(2.f);

        }

    return std::make_tuple(sb0, sb1);
}





