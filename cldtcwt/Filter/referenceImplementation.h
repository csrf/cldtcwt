// Copyright (C) 2013 Timothy Gale
#ifndef REF_IMPLEMENTATION_H
#define REF_IMPLEMENTATION_H

#include <vector>
#include <Eigen/Dense>



// Reference implementations of the various algorithms.  Ease of
// implementation and readability preferred over performance to
// give the best chance of getting the right answer.

Eigen::ArrayXXf convolveRows(const Eigen::ArrayXXf& in, 
                             const std::vector<float>& filter);

Eigen::ArrayXXf convolveCols(const Eigen::ArrayXXf& in, 
                             const std::vector<float>& filter);

Eigen::ArrayXXf decimateConvolveCols(const Eigen::ArrayXXf& in, 
                             const std::vector<float>& filter,
                             bool swapOutputs);

Eigen::ArrayXXf decimateConvolveRows(const Eigen::ArrayXXf& in, 
                             const std::vector<float>& filter,
                             bool swapOutputs);

std::tuple<Eigen::ArrayXXcf, Eigen::ArrayXXcf>
    quadToComplex(const Eigen::ArrayXXf& in);

#endif

