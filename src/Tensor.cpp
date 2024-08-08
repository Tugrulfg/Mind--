#ifndef TENSOR_CPP
#define TENSOR_CPP

#include "../include/Tensor.hpp"

namespace cmind{
    // Constructor for Tensor with given shape
    template<typename T>
    Tensor<T>::Tensor(const std::vector<int>& shape): copied(false), shape_(shape){
        int size = 1;
        for(const int& dim: shape)
            size *= dim;
        std::cout << "Creating Tensor: " << this->shape_ << " " << size << std::endl;
        this->size = size;
        this->step = size / shape[0];
        this->data = (T*)malloc(size*sizeof(T));
    }

    // Constructor for Tensor with given shape
    template<typename T>
    Tensor<T>::Tensor(const Shape& shape): copied(false), shape_(shape){
        int size = 1;
        for(const int& dim: shape)
            size *= dim;
        std::cout << "Creating Tensor: " << this->shape_ << " " << size << std::endl;
        
        this->size = size;
        this->step = size / shape[0];
        this->data = (T*)malloc(size*sizeof(T));
    }

    // Tensor copy contructor for accessing slice of a tensor
    template<typename T>
    Tensor<T>::Tensor(T* data, const std::vector<int>& shape): copied(true), shape_(shape){
        this->data = data;
        int size = 1;
        for(const int& dim: shape)
            size *= dim;
        std::cout << "1.Creating Tensor: " << this->shape_ << " " << size << std::endl;
        
        this->size = size;
        this->step = size / shape[0];
    }

    // Tensor copy contructor for copying whole tensor
    template<typename T>
    Tensor<T>::Tensor(const Tensor<T>& tensor): copied(false), shape_(tensor.shape_){
        this->data = (T*) malloc(tensor.size*sizeof(T));
        std::memcpy(this->data, tensor.data, tensor.size*sizeof(T));
        this->size = tensor.size;
        this->step = tensor.step;
        std::cout << "2.Creating Tensor: " << this->shape_ << " " << size << std::endl;
    }

    // Creates a copy of the tensor
    template<typename T>
    Tensor<T> Tensor<T>::copy()const{
        return Tensor<T>(*this);
    }

    // Returns the shape of the tensor
    template<typename T>
    Shape Tensor<T>::shape()const{
        return this->shape_;
    }

    // Operator overloading for data access through index. r-value
    template<typename T>
    const Tensor<T> Tensor<T>::operator[](const size_t idx)const{
        if(idx>=this->shape_[0] || idx<0){
            std::cerr << "Index out of bounds: " << idx << " " << this->shape_[0] << std::endl;
            abort();
        }

        std::vector<int> shape;
        for(auto it=this->shape_.begin()+1; it!=this->shape_.end(); it++)
            shape.push_back(*it);
        if(shape.size() == 0)
            shape.push_back(1);
        return Tensor<T>(this->data+idx*this->step, shape);
    }

    // Operator overloading for data access through index. l-value
    template<typename T>
    Tensor<T> Tensor<T>::operator[](const size_t idx){
        if(idx>=this->shape_[0] || idx<0){
            std::cerr << "Index out of bounds: " << idx << " " << this->shape_[0] << std::endl;
            abort();
        }
        
        std::vector<int> shape;
        for(auto it=this->shape_.begin()+1; it!=this->shape_.end(); it++)
            shape.push_back(*it);
        if(shape.size() == 0)
            shape.push_back(1);

        return Tensor<T>(this->data+idx*this->step, shape);
    }

    // Operator overloading for assignment
    template<typename T>
    Tensor<T>& Tensor<T>::operator=(const Tensor<T>& other){
        if(this->shape_ != other.shape_){
            std::cerr << "Assignment: Shape mismatch" << std::endl;
            abort();
        }

        memcpy(this->data, other.data, this->size*sizeof(T));
        return *this;
    }

    // Operator overloading for assignment
    template<typename T>
    Tensor<T>& Tensor<T>::operator=(const T val){
        if(this->shape_[0] != 1 && this->shape_.size() != 1){
            std::cerr << "Assignment: Shape mismatch" << std::endl;
            abort();
        }

        *(this->data) = val;
        return *this;
    }

    // Operator overloading for addition
    template<typename T>
    Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const{
        if(this->shape_ != other.shape_){
            std::cerr << "Addition: Shape mismatch" << std::endl;
            abort();
        }
        Tensor<T> copy(*this);

        for(int i=0; i<this->size; i++)
            copy.data[i] += other.data[i];

        return copy;
    }

    // Operator overloading for addition
    template<typename T>
    Tensor<T> Tensor<T>::operator+(const T val) const{
        Tensor<T> copy(*this);
        for(int i=0; i<this->size; i++)
            copy.data[i] += val;
        return copy;
    }

    // Operator overloading for addition
    template<typename T>
    Tensor<T>& Tensor<T>::operator+=(const Tensor<T>& other){
        if(this->shape_ != other.shape_){
            std::cerr << "Addition: Shape mismatch" << std::endl;
            abort();
        }

        for(int i=0; i<this->size; i++)
            this->data[i] += other.data[i];
        return *this;
    }

    // Operator overloading for addition
    template<typename T>
    Tensor<T>& Tensor<T>::operator+=(const T val){
        for(int i=0; i<this->size; i++)
            this->data[i] += val;
        return *this;
    }

    // Operator overloading for subtraction
    template<typename T>
    Tensor<T> Tensor<T>::operator-(const Tensor<T>& other) const{
        if(this->shape_ != other.shape_){
            std::cerr << "Subtraction: Shape mismatch" << std::endl;
            abort();
        }
        Tensor<T> copy(*this);

        for(int i=0; i<this->size; i++)
            copy.data[i] -= other.data[i];

        return copy;
    }

    // Operator overloading for subtraction
    template<typename T>
    Tensor<T> Tensor<T>::operator-(const T val) const{
        Tensor<T> copy(*this);
        for(int i=0; i<this->size; i++)
            copy.data[i] -= val;
        return copy;
    }

    // Operator overloading for subtraction
    template<typename T>
    Tensor<T>& Tensor<T>::operator-=(const Tensor<T>& other){
        if(this->shape_ != other.shape_){
            std::cerr << "Subtraction: Shape mismatch" << std::endl;
            abort();
        }

        for(int i=0; i<this->size; i++)
            this->data[i] -= other.data[i];
        return *this;
    }

    // Operator overloading for subtraction
    template<typename T>
    Tensor<T>& Tensor<T>::operator-=(const T val){
        for(int i=0; i<this->size; i++)
            this->data[i] -= val;
        return *this;
    }

// Operator overloading for multiplication
    template<typename T>
    Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const{
        if(this->shape_ != other.shape_){
            std::cerr << "Multiplication: Shape mismatch" << std::endl;
            abort();
        }
        Tensor<T> copy(*this);

        for(int i=0; i<this->size; i++)
            copy.data[i] *= other.data[i];

        return copy;
    }

    // Operator overloading for multiplication
    template<typename T>
    Tensor<T> Tensor<T>::operator*(const T val) const{
        Tensor<T> copy(*this);
        for(int i=0; i<this->size; i++)
            copy.data[i] *= val;
        return copy;
    }

    // Operator overloading for multiplication
    template<typename T>
    Tensor<T>& Tensor<T>::operator*=(const Tensor<T>& other){
        if(this->shape_ != other.shape_){
            std::cerr << "Multiplication: Shape mismatch" << std::endl;
            abort();
        }

        for(int i=0; i<this->size; i++)
            this->data[i] *= other.data[i];
        return *this;
    }

    // Operator overloading for multiplication
    template<typename T>
    Tensor<T>& Tensor<T>::operator*=(const T val){
        for(int i=0; i<this->size; i++)
            this->data[i] *= val;
        return *this;
    }

    // Operator overloading for division
    template<typename T>
    Tensor<T> Tensor<T>::operator/(const Tensor<T>& other) const{
        if(this->shape_ != other.shape_){
            std::cerr << "Division: Shape mismatch" << std::endl;
            abort();
        }
        Tensor<T> copy(*this);

        for(int i=0; i<this->size; i++){
            if(other.data[i] == 0){
                std::cerr << "Division by zero" << std::endl;
                abort();
            }
            copy.data[i] /= other.data[i];
        }

        return copy;
    }

    // Operator overloading for division
    template<typename T>
    Tensor<T> Tensor<T>::operator/(const T val) const{
        if(val == 0){
            std::cerr << "Division by zero" << std::endl;
            abort();
        }
        Tensor<T> copy(*this);
        for(int i=0; i<this->size; i++)
            copy.data[i] /= val;
        return copy;
    }

    // Operator overloading for division
    template<typename T>
    Tensor<T>& Tensor<T>::operator/=(const Tensor<T>& other){
        if(this->shape_ != other.shape_){
            std::cerr << "Division: Shape mismatch" << std::endl;
            abort();
        }

        for(int i=0; i<this->size; i++){
            if(other.data[i] == 0){
                std::cerr << "Division by zero" << std::endl;
                abort();
            }
            this->data[i] /= other.data[i];
        }
        return *this;
    }

    // Operator overloading for division
    template<typename T>
    Tensor<T>& Tensor<T>::operator/=(const T val){
        if(val == 0){
            std::cerr << "Division by zero" << std::endl;
            abort();
        }
        for(int i=0; i<this->size; i++)
            this->data[i] /= val;
        return *this;
    }

    // Reshaping the tensor
    template<typename T>
    Tensor<T>& Tensor<T>::reshape(const std::vector<int>& new_shape){
        int size = 1;
        for(const int& dim: new_shape)
            size *= dim;
        if(size != this->size){
            std::cerr << "Reshape: Size mismatch" << std::endl;
            abort();
        }
        this->shape_ = Shape(new_shape);
        this->step = size / new_shape[0];
        return copy;
    }

    // Transposing the tensor
    template<typename T>
    Tensor<T>& Tensor<T>::transpose(){
        // TODO: Implement transpose
        return *this;
    }

    // Slicing the tensor
    template<typename T>
    Tensor<T>& Tensor<T>::slice(const std::vector<int>& start, const std::vector<int>& end) const{
        // TODO: Implement slice
        return *this;
    }

    // Sum of the tensor
    template<typename T>
    T Tensor<T>::sum() const{
        T sum = 0;
        for(int i=0; i<this->size; i++)
            sum += this->data[i];
        return sum;
    }

    // Mean of the tensor
    template<typename T>
    float Tensor<T>::mean() const{
        return this->sum() / this->size;
    }

    // Min of the tensor
    template<typename T>
    T Tensor<T>::min() const{
        T min = this->data[0];
        for(int i=1; i<this->size; i++)
            if(this->data[i] < min)
                min = this->data[i];
        return min;
    }

    // Max of the tensor
    template<typename T>
    T Tensor<T>::max() const{
        T max = this->data[0];
        for(int i=1; i<this->size; i++)
            if(this->data[i] > max)
                max = this->data[i];
        return max;
    }

    // Destructor of Tensor
    template<typename T>
    Tensor<T>::~Tensor(){
        if(!this->copied && this->data != nullptr){
            std::cout << "Deleting Tensor: " << this->shape_ << std::endl;
            free(this->data);
            this->data = nullptr;
        }
    }

    // Overloading the << operator
    template<typename T>
    std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor){
        if(tensor.shape_.size()==1){
            os << "[ "; 
            for(int i=0; i<tensor.shape_[0]; i++)
                os << tensor.data[i] << " ";
            os << "]" << std::endl;
        }
        else if(tensor.shape_.size()==2){
            os << "Tensor" << std::endl;
            os << "[" << std::endl;
            for(int i=0; i<tensor.shape_[0]; i++){
                os << i << ".   [ ";
                for(int j=0; j<tensor.shape_[1]; j++)
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