#ifndef MATH_HPP
#define MATH_HPP

#include "../include/Mind++.hpp"

namespace cmind{

    // Return the absolute difference of the given tensors
    Tensor<float> abs_dif(const Tensor<float>& input1, const Tensor<float>& input2);

    // Square root of the given tensor
    Tensor<float> sqrt(const Tensor<float>& input);

}

#endif