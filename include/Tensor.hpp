#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
// #include "C++Mind.hpp"
#include <stdlib.h>

using namespace std;

template<typename T>
class Tensor{

    public:
        // Constructor for Tensor with given shape and data type
        Tensor(vector<int> shape);

        // operator[](int idx);

        // Destructor of Tensor
        ~Tensor();
    
    private:
        vector<int> shape;
        T* data;
};

// Template class member function definitions
#include "../src/Tensor.cpp"


#endif