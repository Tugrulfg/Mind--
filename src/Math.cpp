#ifndef MATH_CPP
#define MATH_CPP

#include "../include/Math.hpp"
#include <cmath>

namespace cmind{

    // Return the absolute value of the tensor
    template<typename T>
    Tensor<T> abs(const Tensor<T>& input){
        Tensor<T> output(input.shape());
        const T* in = input.data();
    
        for(size_t i=0; i<input.size(); i++){
            if(in[i] < 0)
                output[i] = -in[i];
            else
                output[i] = in[i];
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

    // Power of the given tensor
    template<typename T>
    Tensor<T> power(const Tensor<T>& input1, const int degree){
        if(input1.size() != 1){
            std::cout << "power: Input tensor must be of size 1" << std::endl;
            abort();
        }
        Tensor<T> output({1});
        output = 1;
        T in = *input1.data();
        for(size_t i=1; i<=degree; i++)
            output *= in;
        return output;
    }

    // Matrix multiplication
    template<typename T>
    Tensor<T> mat_mul(const Tensor<T>& input1, const Tensor<T>& input2){
        if(input1.shape()[1] != input2.shape()[0] || input1.shape().size() != 2 || input2.shape().size() != 2){
            std::cout << "mat_mul: Shapes do not match" << std::endl;
            abort();
        }

        Tensor<T> output({input1.shape()[0], input2.shape()[1]});
        output.fill(0.0);

        for(size_t i=0; i<input1.shape()[0]; i++){
            for(size_t j=0; j<input2.shape()[1]; j++){
                for(size_t k=0; k<input1.shape()[1]; k++)
                    output[i][j] += input1[i][k]*input2[k][j];
            }
        }
        return output;
    }

    // Logarithm of the given tensor
    template<typename T>
    Tensor<T> log(const Tensor<T>& input){
        Tensor<T> output(input.shape());
        const T* in = input.data();
        T* out = output.data(); 
        for(size_t i=0; i<input.size(); i++)
            out[i] = std::log(in[i]);
        return output;
    }

    // Exponential of the given tensor
    template<typename T>
    Tensor<T> exp(const Tensor<T>& input){
        Tensor<T> output(input.shape());
        const T* in = input.data();
        T* out = output.data(); 
        for(size_t i=0; i<input.size(); i++)
            out[i] = std::exp(in[i]);
        return output;
    }

}

#endif