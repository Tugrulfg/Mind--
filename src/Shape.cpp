#include "../include/Shape.hpp"

// Shape object constructor
Shape::Shape(const std::vector<int>& shape): shape(shape){
    std::cout << "Creating Shape: " << this->shape.size() << std::endl;
}

// Overloading the [] operator
const int& Shape::operator[](const int idx)const{
    if(idx>=this->shape.size() || idx<0){
        std::cerr << "Index out of bounds" << std::endl;
        abort();
    }
        
    return this->shape[idx];
}

// Returns the size of the shape
size_t Shape::size()const{
    return this->shape.size();
}

// Returns the begin iterator
std::vector<int>::const_iterator Shape::begin()const{ 
    return this->shape.begin(); 
}

// Returns the end iterator
std::vector<int>::const_iterator Shape::end()const{ 
    return this->shape.end(); 
}

// Returns the data pointer
const int* Shape::data()const{ 
    return this->shape.data(); 
}

// Overloading the << operator
// std::ostream& operator<<(std::ostream& os, Shape& shape){
//     os << "[ ";
//     for(const int& dim: shape)
//         os << dim << " ";
//     os << "]";
//     return os;
// }

// Overloading the << operator
std::ostream& operator<<(std::ostream& os, const Shape shape){
    os << "[ ";
    for(const int& dim: shape)
        os << dim << " ";
    os << "]";
    return os;
}