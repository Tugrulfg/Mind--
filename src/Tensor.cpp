#ifndef TENSOR_CPP
#define TENSOR_CPP

#include "../include/Tensor.hpp"

// Constructor for Tensor with given shape and data type
template<typename T>
Tensor<T>::Tensor(vector<int> shape){
    this->shape = shape;

    int size = 0;
    for(const int& dim: shape)
        size += dim;

    this->data = (T*)malloc(size*sizeof(T));
}

// Destructor of Tensor
template<typename T>
Tensor<T>::~Tensor(){
    free(this->data);
}

#endif