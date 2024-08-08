#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include "Shape.hpp"
#include <stdlib.h>
#include <ostream>
#include <iostream>
#include <cstdlib>
#include <cstring>

namespace cmind{

    template<typename T>
    class Tensor{
        public:
            // Constructor for Tensor with given shape
            Tensor(const std::vector<size_t>& shape);

            // Constructor for Tensor with given shape
            Tensor(const Shape& shape);

            // Creates a copy of the tensor
            Tensor<T> copy()const;

            // Returns the data pointer
            T* data();

            // Returns the shape of the tensor
            Shape shape()const;

            // Operator overloading for data access through index. r-value
            const Tensor<T> operator[](const size_t idx)const;

            // Operator overloading for data access through index. l-value
            Tensor<T> operator[](const size_t idx);

            // Operator overloading for assignment
            Tensor<T>& operator=(const Tensor<T>& other);
            Tensor<T>& operator=(const T val);

            // Operator overloading for addition
            Tensor<T> operator+(const Tensor<T>& other) const;
            Tensor<T> operator+(const T val) const;
            Tensor<T>& operator+=(const Tensor<T>& other);
            Tensor<T>& operator+=(const T val);

            // Operator overloading for subtraction
            Tensor<T> operator-(const Tensor<T>& other) const;
            Tensor<T> operator-(const T val) const;
            Tensor<T>& operator-=(const Tensor<T>& other);
            Tensor<T>& operator-=(const T val);

            // Operator overloading for multiplication
            Tensor<T> operator*(const Tensor<T>& other) const;
            Tensor<T> operator*(const T val) const;
            Tensor<T>& operator*=(const Tensor<T>& other);
            Tensor<T>& operator*=(const T val);

            // Operator overloading for division
            Tensor<T> operator/(const Tensor<T>& other) const;
            Tensor<T> operator/(const T val) const;
            Tensor<T>& operator/=(const Tensor<T>& other);
            Tensor<T>& operator/=(const T val);

            // Utility functions
            Tensor<T>& reshape(const std::vector<size_t>& new_shape);
            Tensor<T>& transpose();
            Tensor<T>& slice(const std::vector<int>& start, const std::vector<int>& end) const;
            T sum() const;
            float mean() const;
            T min() const;
            T max() const;

            // Destructor of Tensor
            ~Tensor();

            // Overloading the << operator
            template <typename U>
            friend std::ostream& operator<<(std::ostream& os, const Tensor<U>& tensor);

        private:
            // Tensor copy contructor for accessing slice of a tensor
            Tensor(T* data, const std::vector<size_t>& shape);

            // Tensor copy contructor for copying whole tensor
            Tensor(const Tensor<T>& tensor);

            T* data_;
            const bool copied; // Checks if the tensor is a copy or original
            size_t size; // Total number of the elements in the tensor
            size_t step = 1; // Step size distance between each index
            Shape shape_;
    };

}
// Template class member function definitions
#include "../src/Tensor.cpp"


#endif