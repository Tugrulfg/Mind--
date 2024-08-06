#ifndef TENSOR_CPP
#define TENSOR_CPP

#include "../include/Tensor.hpp"

// Constructor for Tensor with given shape and data type
template<typename T>
Tensor<T>::Tensor(const std::vector<int>& shape): copy(false), shape(shape){
    int size = 1;
    std::cout << "Creating Tensor: " << this->shape << std::endl;
    
    this->step = size / shape[0];
    this->data = (T*)malloc(size*sizeof(T));
}

// Tensor copy contructor
template<typename T>
Tensor<T>::Tensor(T* data, const std::vector<int>& shape): copy(true), shape(shape){
    this->data = data;
    
    for(const int& dim: shape)
        this->step *= dim;
    this->step /= shape[0];
}

// Operator overloading for data access through index. r-value
template<typename T>
const Tensor<T>& Tensor<T>::operator[](size_t idx)const{
    if(idx>=this->shape[0] || idx<0){
        std::cerr << "Index out of bounds" << std::endl;
        abort();
    }

    return Tensor<T>(this->data+idx*this->step, {this->shape.data()+sizeof(int)});
}

// Operator overloading for data access through index. l-value
template<typename T>
Tensor<T>& Tensor<T>::operator[](size_t idx){
    if(idx>=shape[0] || idx<0){
        std::cerr << "Index out of bounds" << std::endl;
        abort();
    }

    return Tensor<T>(this->data+idx*this->step, {this->shape.data()+sizeof(int)});
}

// Destructor of Tensor
template<typename T>
Tensor<T>::~Tensor(){
    if(!this->copy)
        free(this->data);
}

#endif