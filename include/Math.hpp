#ifndef MATH_HPP
#define MATH_HPP

#include "../include/Mind++.hpp"

namespace cmind{

    // Return the absolute value of the tensor
    template<typename T>
    Tensor<T> abs(const Tensor<T>& input);

    // Square root of the given tensor
    template<typename T>
    Tensor<T> sqrt(const Tensor<T>& input);

    // Factorial of the given tensor
    template<typename T>
    Tensor<T> factorial(const Tensor<T>& input);

    // Power of the given tensor
    template<typename T>
    Tensor<T> power(const Tensor<T>& input1, const Tensor<T>& degree);

    // Power of the given tensor
    template<typename T>
    Tensor<T> power(const Tensor<T>& input1, const int degree);

    // Matrix multiplication
    template<typename T>
    Tensor<T> mat_mul(const Tensor<T>& input1, const Tensor<T>& input2);

}

#include "../src/Math.cpp"

#endif