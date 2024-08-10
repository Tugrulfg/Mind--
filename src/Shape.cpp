#include "../include/Shape.hpp"

namespace cmind{
    // Shape object constructor
    Shape::Shape(const std::vector<size_t>& shape): shape(shape){
        // std::cout << "Creating Shape: " << this->shape.size() << std::endl;
    }

    // Shape object constructor
    // Shape::Shape(): shape({}){
        
    // }

    // Overloading the [] operator
    const size_t& Shape::operator[](const size_t idx)const{
        if(idx>=this->shape.size()){
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
    std::vector<size_t>::const_iterator Shape::begin()const{ 
        return this->shape.begin(); 
    }

    // Returns the end iterator
    std::vector<size_t>::const_iterator Shape::end()const{ 
        return this->shape.end(); 
    }

    // Returns the data pointer
    const size_t* Shape::data()const{ 
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
        for(const size_t& dim: shape)
            os << dim << " ";
        os << "]";
        return os;
    }

    // Overloading the == operator
    bool Shape::operator==(const Shape& other)const{
        if(this->shape.size() != other.shape.size())
            return false;

        for(size_t i=0; i<this->shape.size(); i++){
            if(this->shape[i] != other.shape[i])
                return false;
        }
        return true;
    }

    // Overloading the != operator
    bool Shape::operator!=(const Shape& other)const{
        return !(*this == other);
    }

}