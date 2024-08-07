#ifndef TENSOR_CPP
#define TENSOR_CPP

#include "../include/Tensor.hpp"

namespace cmind{
    // Constructor for Tensor with given shape
    template<typename T>
    Tensor<T>::Tensor(const std::vector<int>& shape): copied(false), shape(shape){
        int size = 0;
        for(const int& dim: shape)
            size += dim;
        std::cout << "Creating Tensor: " << this->shape << " " << size << std::endl;
        this->size = size;
        this->step = size / shape[0];
        this->data = (T*)malloc(size*sizeof(T));
    }

    // Constructor for Tensor with given shape
    template<typename T>
    Tensor<T>::Tensor(const Shape& shape): copied(false), shape(shape){
        int size = 0;
        for(const int& dim: shape)
            size += dim;
        std::cout << "Creating Tensor: " << this->shape << " " << size << std::endl;
        
        this->size = size;
        this->step = size / shape[0];
        this->data = (T*)malloc(size*sizeof(T));
    }

    // Tensor copy contructor for accessing slice of a tensor
    template<typename T>
    Tensor<T>::Tensor(T* data, const std::vector<int>& shape): copied(true), shape(shape){
        this->data = data;
        int size = 0;
        for(const int& dim: shape)
            size += dim;
        std::cout << "1.Creating Tensor: " << this->shape << " " << size << std::endl;
        
        this->size = size;
        this->step = size / shape[0];
    }

    // Tensor copy contructor for copying whole tensor
    template<typename T>
    Tensor<T>::Tensor(T* data, const Shape& shape): copied(true), shape(shape){
        this->data = data;
        int size = 0;
        for(const int& dim: shape)
            size += dim;
        std::cout << "2.Creating Tensor: " << this->shape << " " << size << std::endl;
        
        this->size = size;
        this->step = size / shape[0];
    }

    // Creates a copy of the tensor
    template<typename T>
    Tensor<T> Tensor<T>::copy()const{
        T* copy = (T*)malloc(this->size*sizeof(T));
        std::memcpy(this->data, copy, this->size*sizeof(T));
        return Tensor<T>(copy, this->shape);
    }

    // Operator overloading for data access through index. r-value
    template<typename T>
    const Tensor<T> Tensor<T>::operator[](const size_t idx)const{
        if(idx>=this->shape[0] || idx<0){
            std::cerr << "Index out of bounds" << std::endl;
            abort();
        }

        std::vector<int> shape;
        for(auto it=this->shape.begin()+1; it!=this->shape.end(); it++)
            shape.push_back(*it);
        if(shape.size() == 0)
            shape.push_back(1);
        return Tensor<T>(this->data+idx*this->step, shape);
    }

    // Operator overloading for data access through index. l-value
    template<typename T>
    Tensor<T> Tensor<T>::operator[](const size_t idx){
        if(idx>=shape[0] || idx<0){
            std::cerr << "Index out of bounds: " << idx << " " << shape[0] << std::endl;
            abort();
        }
        
        std::vector<int> shape;
        for(auto it=this->shape.begin()+1; it!=this->shape.end(); it++)
            shape.push_back(*it);
        if(shape.size() == 0)
            shape.push_back(1);
        return Tensor<T>(this->data+idx*this->step, shape);
    }

    // Operator overloading for assignment
    template<typename T>
    Tensor<T>& Tensor<T>::operator=(const Tensor<T>& other){
        if(this->shape != other.shape){
            std::cerr << "Shape mismatch" << std::endl;
            abort();
        }

        memcpy(this->data, other.data, this->size*sizeof(T));
        return *this;
    }

    // Operator overloading for assignment
    template<typename T>
    Tensor<T>& Tensor<T>::operator=(const T val){
        if(this->shape[0] != 1 && this->shape.size() != 1){
            std::cerr << "Shape mismatch" << std::endl;
            abort();
        }

        *(this->data) = val;
        return *this;
    }

    // Destructor of Tensor
    template<typename T>
    Tensor<T>::~Tensor(){
        if(!this->copied && this->data != nullptr){
            std::cout << "Deleting Tensor: " << this->shape << std::endl;
            free(this->data);
            this->data = nullptr;
        }
    }

    // Overloading the << operator
    template<typename T>
    std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor){
        if(tensor.shape.size()==1){
            os << "[ "; 
            for(int i=0; i<tensor.shape[0]; i++)
                os << tensor.data[i] << " ";
            os << "]" << std::endl;
        }
        else if(tensor.shape.size()==2){
            os << "Tensor" << std::endl;
            os << "[" << std::endl;
            for(int i=0; i<tensor.shape[0]; i++){
                os << "   [ ";
                for(int j=0; j<tensor.shape[1]; j++)
                    os << tensor.data[i*tensor.step+j] << " ";
                os << "]" << std::endl;
            }   
            os << "]" << std::endl;
        }
        else{
            os << "Couldn't print the tensors with more than 2 dimensions" << std::endl;
        }
        return os;
    }

}
#endif