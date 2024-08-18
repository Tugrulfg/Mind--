#ifndef UTILS_HPP
#define UTILS_HPP

#include "Mind++.hpp"

namespace cmind{

    // Split the line based on the delimiter
    std::vector<std::string> split(const std::string& line, char delimiter = ' ');

    // Random number generator
    size_t random(size_t min, size_t max);
}

#endif