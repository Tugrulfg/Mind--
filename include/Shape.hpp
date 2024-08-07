#ifndef SHAPE_HPP
#define SHAPE_HPP

#include <vector>
#include <ostream>
#include <iostream>
#include <cstdlib>

namespace cmind{
    // Class for representing the shapes of the tensors
    class Shape{
        public:
            // Shape object constructor
            Shape(const std::vector<int>& shape);

            // Overloading the [] operator
            const int& operator[](const size_t idx)const;

            // Returns the size of the shape
            size_t size()const;

            // Returns the begin iterator
            std::vector<int>::const_iterator begin()const;

            // Returns the end iterator
            std::vector<int>::const_iterator end()const;

            // Returns the data pointer
            const int* data()const;

            // Overloading the == operator
            bool operator==(const Shape& other)const;

            // Overloading the != operator
            bool operator!=(const Shape& other)const;
        private:
            // Shape vector
            const std::vector<int> shape;
    };

    // Overloading the << operator
    // std::ostream& operator<<(std::ostream& os, Shape& shape);

    // Overloading the << operator
    std::ostream& operator<<(std::ostream& os, const Shape shape);
}

#endif