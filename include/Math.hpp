#ifndef MATH_HPP
#define MATH_HPP

#include "../include/Mind++.hpp"

namespace cmind{

    // Return the absolute difference of the given tensors
    template<typename T>
    Tensor<T> abs_dif(const Tensor<T>& input1, const Tensor<T>& input2);

    // Square root of the given tensor
    template<typename T>
    Tensor<T> sqrt(const Tensor<T>& input);

    // Factorial of the given tensor
    template<typename T>
    Tensor<T> factorial(const Tensor<T>& input);

    // Power of the given tensor
    template<typename T>
    Tensor<T> power(const Tensor<T>& input1, const Tensor<T>& degree);

}

#include "../src/Math.cpp"

#endif