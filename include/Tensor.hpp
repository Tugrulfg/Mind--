#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include "Shape.hpp"
#include <stdlib.h>
#include <ostream>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <random>

namespace cmind{

    template<typename T>
    class Tensor{
        public:
            // Constructor for Tensor with given shape
            Tensor(const std::vector<size_t>& shape);

            // Constructor for Tensor with given shape
            Tensor(const Shape& shape);

            // Tensor copy contructor for copying whole tensor
            Tensor(const Tensor<T>& tensor);

            // Tensor copy contructor for accessing slice of a tensor
            Tensor(T* data, const std::vector<size_t>& shape);

            // Default constructor
            Tensor();

            // Fill the tensor with the given value
            void fill(const T val);

            // Fill randomly the tensor
            void randomize();

            // Creates a copy of the tensor
            Tensor<T> copy()const;

            // Returns the data pointer
            const T* data()const;

            // Returns the first value store
            T first()const;

            // Returns the value at the given index
            T at(const Shape shape)const;

            // Returns the shape of the tensor
            Shape shape()const;

            // Returns the size of the tensor
            size_t size()const;

            // Returns if the tensor's all the elements are negative
            bool all_negative()const;

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
            Tensor<T> operator-() const;

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

            // Operator overloading for comparison
            bool operator<(const Tensor<T>& other) const;
            bool operator<(const T val) const;
            bool operator<=(const Tensor<T>& other) const;
            bool operator<=(const T val) const;
            bool operator>(const Tensor<T>& other) const;
            bool operator>(const T val) const;
            bool operator>=(const Tensor<T>& other) const;
            bool operator>=(const T val) const;
            bool operator==(const Tensor<T>& other) const;
            bool operator==(const T val) const;
            bool operator!=(const Tensor<T>& other) const;
            bool operator!=(const T val) const;

            // Utility functions
            Tensor<T>& reshape(const std::vector<size_t>& new_shape);
            Tensor<T>& transpose();
            Tensor<T>& slice(const std::vector<int>& start, const std::vector<int>& end) const;
            Tensor<T> flatten()const;

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

            T* data_;
            const bool copied; // Checks if the tensor is a copy or original
            size_t size_; // Total number of the elements in the tensor
            size_t step = 1; // Step size distance between each index
            Shape shape_;
    };


    template<typename T>
    Tensor<T> operator-(T val, const Tensor<T>& tensor);

    template<typename T>
    Tensor<T> operator+(T val, const Tensor<T>& tensor);

    template<typename T>
    Tensor<T> operator*(T val, const Tensor<T>& tensor);
}
// Template class member function definitions
#include "../src/Tensor.cpp"


#endif