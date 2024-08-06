#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include "Shape.hpp"
#include <stdlib.h>
#include <iostream>
#include <cstdlib>

using namespace std;

template<typename T>
class Tensor{

    public:
        // Constructor for Tensor with given shape and data type
        Tensor(const std::vector<int>& shape);

        // Tensor copy contructor
        Tensor(T* data, const std::vector<int>& shape);

        // Operator overloading for data access through index. r-value
        const Tensor<T>& operator[](size_t idx)const;

        // Operator overloading for data access through index. l-value
        Tensor<T>& operator[](size_t idx);

        // Destructor of Tensor
        ~Tensor();

        const Shape shape;
    private:
        T* data;
        const bool copy; // Checks if the tensor is a copy or original
        int step = 1; // Step size distance between each index
};

// Template class member function definitions
#include "../src/Tensor.cpp"


#endif