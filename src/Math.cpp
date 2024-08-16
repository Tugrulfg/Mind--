#ifndef MATH_CPP
#define MATH_CPP

#include "../include/Math.hpp"
#include <cmath>


namespace cmind{

    // Return the absolute difference of the given tensors
    template<typename T>
    Tensor<T> abs_dif(const Tensor<T>& input1, const Tensor<T>& input2){
        if(input1.shape() != input2.shape()){
            std::cout << "abs_dif: Shapes do not match" << std::endl;
            abort();
        }
        Tensor<T> output(input1.shape());
        const T* in1 = input1.data();
        const T* in2 = input2.data();
        for(size_t i=0; i<input1.size(); i++){
            if(in1[i] < in2[i])
                output[i] = in2[i]-in1[i];
            else
                output[i] = in1[i]-in2[i];
        }
        return output;
    }

    // Square root of the given tensor
    template<typename T>
    Tensor<T> sqrt(const Tensor<T>& input){
        Tensor<T> output(input.shape());
        const T* in = input.data();

        for(size_t i=0; i<input.size(); i++)
            output[i] = std::sqrt(in[i]);

        return output;
    }

    // Factorial of the given tensor
    template<typename T>
    Tensor<T> factorial(const Tensor<T>& input){
        if(input.size() != 1){
            std::cout << "factorial: Input tensor must be of size 1" << std::endl;
            abort();
        }
        Tensor<T> output({1});
        output = 1;
        T in = *input.data();


        for(size_t i=1; i<=in; i++)
            output *= i;

        return output;
    }

    // Power of the given tensor
    template<typename T>
    Tensor<T> power(const Tensor<T>& input1, const Tensor<T>& degree){
        if(input1.size() != 1){
            std::cout << "power: Input tensor must be of size 1" << std::endl;
            abort();
        }
        if(degree.size() != 1){
            std::cout << "power: Degree tensor must be of size 1" << std::endl;
            abort();
        }
        Tensor<T> output({1});
        output = 1;
        T in = *input1.data();
        T deg = *degree.data();
        for(size_t i=1; i<=deg; i++)
            output *= in;
        return output;
    }



}

#endif