#ifndef TENSOR_CPP
#define TENSOR_CPP

#include "../include/Tensor.hpp"

namespace cmind{
    // Constructor for Tensor with given shape
    template<typename T>
    Tensor<T>::Tensor(const std::vector<size_t>& shape): copied(false), shape_(shape){
        size_t size = 1;
        for(const size_t& dim: shape)
            size *= dim;
        // std::cout << "Creating Tensor: " << this->shape_ << " " << size << std::endl;
        this->size_ = size;
        this->step = size / shape[0];
        this->data_ = (T*)malloc(size*sizeof(T));
    }

    // Constructor for Tensor with given shape
    template<typename T>
    Tensor<T>::Tensor(const Shape& shape): copied(false), shape_(shape){
        size_t size = 1;
        for(const size_t& dim: shape)
            size *= dim;
        // std::cout << "Creating Tensor: " << this->shape_ << " " << size << std::endl;
        
        this->size_ = size;
        this->step = size / shape[0];
        this->data_ = (T*)malloc(size*sizeof(T));
    }

    // Tensor copy contructor for accessing slice of a tensor
    template<typename T>
    Tensor<T>::Tensor(T* data, const std::vector<size_t>& shape): copied(true), shape_(shape){
        this->data_ = data;
        size_t size = 1;
        for(const size_t& dim: shape)
            size *= dim;
        // std::cout << "1.Creating Tensor: " << this->shape_ << " " << size << std::endl;
        
        this->size_ = size;
        this->step = size / shape[0];
    }

    // Tensor copy contructor for copying whole tensor
    template<typename T>
    Tensor<T>::Tensor(const Tensor<T>& tensor): copied(false), shape_(tensor.shape_){
        this->data_ = (T*) malloc(tensor.size_*sizeof(T));
        std::memcpy(this->data_, tensor.data_, tensor.size_*sizeof(T));
        this->size_ = tensor.size_;
        this->step = tensor.step;
        // std::cout << "2.Creating Tensor: " << this->shape_ << " " << size << std::endl;
    }

    // Default constructor
    template<typename T>
    Tensor<T>::Tensor(): copied(false), shape_({0}){

    }

    // Fill the tensor with the given value
    template<typename T>
    void Tensor<T>::fill(const T val){
        std::fill(data_, data_+size_, val);
    }

    // Fill randomly the tensor
    template<typename T>
    void Tensor<T>::randomize(){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0, 1);
        for(size_t i=0; i<size_; i++)
            this->data_[i] = dis(gen);
    }

    // Creates a copy of the tensor
    template<typename T>
    Tensor<T> Tensor<T>::copy()const{
        return Tensor<T>(*this);
    }

    // Returns the data pointer
    template<typename T>
    const T* Tensor<T>::data()const{
        return this->data_;
    }

    // Returns the first value store
    template<typename T>
    T Tensor<T>::first()const{
        return this->data_[0];
    }

    // Returns the value at the given index
    template<typename T>
    T Tensor<T>::at(const Shape shape)const{
        if(shape.size() != this->shape_.size()){
            std::cerr << "Shape mismatch: " << shape << " " << this->shape_ << std::endl;
            abort();
        }
        Tensor<T> tensor;
        for(size_t s: shape)
            tensor = tensor[s];
        return tensor.first();
    }

    // Returns the shape of the tensor
    template<typename T>
    Shape Tensor<T>::shape()const{
        return this->shape_;
    }

    // Returns the size of the tensor
    template<typename T>
    size_t Tensor<T>::size()const{
        return this->size_;
    }

    // Returns if the tensor's all the elements are negative
    template<typename T>
    bool Tensor<T>::all_negative()const{
        for(size_t i=0; i<size_; i++)
            if(this->data_[i] >= 0)
                return false;
        return true;
    }

    // Operator overloading for data access through index. r-value
    template<typename T>
    const Tensor<T> Tensor<T>::operator[](const size_t idx)const{
        if(idx>=this->shape_[0]){
            std::cerr << "Index out of bounds: " << idx << " " << this->shape_[0] << std::endl;
            abort();
        }

        std::vector<size_t> shape;
        for(auto it=this->shape_.begin()+1; it!=this->shape_.end(); it++)
            shape.push_back(*it);
        if(shape.size() == 0)
            shape.push_back(1);
        return Tensor<T>(this->data_+idx*this->step, shape);
    }

    // Operator overloading for data access through index. l-value
    template<typename T>
    Tensor<T> Tensor<T>::operator[](const size_t idx){
        if(idx>=this->shape_[0]){
            std::cerr << "Index out of bounds: " << idx << " " << this->shape_[0] << std::endl;
            abort();
        }
        
        std::vector<size_t> shape;
        for(auto it=this->shape_.begin()+1; it!=this->shape_.end(); it++)
            shape.push_back(*it);
        if(shape.size() == 0)
            shape.push_back(1);

        return Tensor<T>(this->data_+idx*this->step, shape);
    }

    // Operator overloading for assignment
    template<typename T>
    Tensor<T>& Tensor<T>::operator=(const Tensor<T>& other){
        if(this->shape_ != other.shape_){
            std::cerr << "Assignment: Shape mismatch" << this->shape_ << " " << other.shape_ << std::endl;
            abort();
        }

        memcpy(this->data_, other.data_, this->size_*sizeof(T));
        return *this;
    }

    // Operator overloading for assignment
    template<typename T>
    Tensor<T>& Tensor<T>::operator=(const T val){
        if(this->shape_[0] != 1 && this->shape_.size() != 1){
            std::cerr << "Assignment: Shape mismatch: " << this->shape_ << std::endl;
            abort();
        }

        *(this->data_) = val;
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

        for(size_t i=0; i<this->size_; i++)
            copy.data_[i] += other.data_[i];

        return copy;
    }

    // Operator overloading for addition
    template<typename T>
    Tensor<T> Tensor<T>::operator+(const T val) const{
        Tensor<T> copy(*this);
        for(size_t i=0; i<this->size_; i++)
            copy.data_[i] += val;
        return copy;
    }

    // Operator overloading for addition
    template<typename T>
    Tensor<T>& Tensor<T>::operator+=(const Tensor<T>& other){
        if(this->shape_ != other.shape_){
            std::cerr << "Addition: Shape mismatch" << std::endl;
            abort();
        }

        for(size_t i=0; i<this->size_; i++)
            this->data_[i] += other.data_[i];
        return *this;
    }

    // Operator overloading for addition
    template<typename T>
    Tensor<T>& Tensor<T>::operator+=(const T val){
        for(size_t i=0; i<this->size_; i++)
            this->data_[i] += val;
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

        for(size_t i=0; i<this->size_; i++)
            copy.data_[i] -= other.data_[i];

        return copy;
    }

    // Operator overloading for subtraction
    template<typename T>
    Tensor<T> Tensor<T>::operator-(const T val) const{
        Tensor<T> copy(*this);
        for(size_t i=0; i<this->size_; i++)
            copy.data_[i] -= val;
        return copy;
    }

    // Operator overloading for subtraction
    template<typename T>
    Tensor<T>& Tensor<T>::operator-=(const Tensor<T>& other){
        if(this->shape_ != other.shape_){
            std::cerr << "Subtraction: Shape mismatch" << std::endl;
            abort();
        }

        for(size_t i=0; i<this->size_; i++)
            this->data_[i] -= other.data_[i];
        return *this;
    }

    // Operator overloading for subtraction
    template<typename T>
    Tensor<T>& Tensor<T>::operator-=(const T val){
        for(size_t i=0; i<this->size_; i++)
            this->data_[i] -= val;
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

        for(size_t i=0; i<this->size_; i++)
            copy.data_[i] *= other.data_[i];

        return copy;
    }

    // Operator overloading for multiplication
    template<typename T>
    Tensor<T> Tensor<T>::operator*(const T val) const{
        Tensor<T> copy(*this);
        for(size_t i=0; i<this->size_; i++)
            copy.data_[i] *= val;
        return copy;
    }

    // Operator overloading for multiplication
    template<typename T>
    Tensor<T>& Tensor<T>::operator*=(const Tensor<T>& other){
        if(this->shape_ != other.shape_){
            std::cerr << "Multiplication: Shape mismatch" << std::endl;
            abort();
        }

        for(size_t i=0; i<this->size_; i++)
            this->data_[i] *= other.data_[i];
        return *this;
    }

    // Operator overloading for multiplication
    template<typename T>
    Tensor<T>& Tensor<T>::operator*=(const T val){
        for(size_t i=0; i<this->size_; i++)
            this->data_[i] *= val;
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

        for(size_t i=0; i<this->size_; i++){
            if(other.data_[i] == 0){
                std::cerr << "Division by zero" << std::endl;
                abort();
            }
            copy.data_[i] /= other.data_[i];
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
        for(size_t i=0; i<this->size_; i++)
            copy.data_[i] /= val;
        return copy;
    }

    // Operator overloading for division
    template<typename T>
    Tensor<T>& Tensor<T>::operator/=(const Tensor<T>& other){
        if(this->shape_ != other.shape_){
            std::cerr << "Division: Shape mismatch" << std::endl;
            abort();
        }

        for(size_t i=0; i<this->size_; i++){
            if(other.data_[i] == 0){
                std::cerr << "Division by zero" << std::endl;
                abort();
            }
            this->data_[i] /= other.data_[i];
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
        for(size_t i=0; i<this->size_; i++)
            this->data_[i] /= val;
        return *this;
    }

    // Reshaping the tensor
    template<typename T>
    Tensor<T>& Tensor<T>::reshape(const std::vector<size_t>& new_shape){
        size_t size = 1;
        for(const size_t& dim: new_shape)
            size *= dim;
        if(size != this->size_){
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
        for(size_t i=0; i<this->size_; i++)
            sum += this->data_[i];
        return sum;
    }

    // Mean of the tensor
    template<typename T>
    float Tensor<T>::mean() const{
        return this->sum() / this->size_;
    }

    // Min of the tensor
    template<typename T>
    T Tensor<T>::min() const{
        T min = this->data_[0];
        for(size_t i=1; i<this->size_; i++)
            if(this->data_[i] < min)
                min = this->data_[i];
        return min;
    }

    // Max of the tensor
    template<typename T>
    T Tensor<T>::max() const{
        T max = this->data_[0];
        for(size_t i=1; i<this->size_; i++)
            if(this->data_[i] > max)
                max = this->data_[i];
        return max;
    }

    // Destructor of Tensor
    template<typename T>
    Tensor<T>::~Tensor(){
        if(!this->copied && this->data_ != nullptr){
            // std::cout << "Deleting Tensor: " << this->shape_ << std::endl;
            free(this->data_);
            this->data_ = nullptr;
        }
    }

    // Overloading the << operator
    template<typename T>
    std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor){
        if(tensor.shape_.size()==1){
            if(tensor.shape_[0] == 1)
                os << tensor.data_[0];
            else{
                os << "[ "; 
                for(size_t i=0; i<tensor.shape_[0]; i++)
                    os << tensor.data_[i] << " ";
                os << "]" << std::endl;
            }
        }
        else if(tensor.shape_.size()==2){
            os << "Tensor" << std::endl;
            os << "[" << std::endl;
            for(size_t i=0; i<tensor.shape_[0]; i++){
                os << i << ".   [ ";
                for(size_t j=0; j<tensor.shape_[1]; j++)
                    os << tensor.data_[i*tensor.step+j] << " ";
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