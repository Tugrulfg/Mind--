#ifndef MATH_HPP
#define MATH_HPP

#include "../include/Mind++.hpp"

namespace cmind{

    // Return the absolute value of the given tensors
    Tensor<float> abs(const Tensor<float>& input1, const Tensor<float>& input2);

}

#endif