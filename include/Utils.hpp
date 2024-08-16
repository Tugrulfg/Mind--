#ifndef UTILS_HPP
#define UTILS_HPP

#include "Mind++.hpp"

namespace cmind{

    // Convert the input to polynomial features
    std::vector<float> poly_features(const std::vector<double>& input, int degree);

    // Helper function to recursively generate polynomial features
    void generatePolynomialFeatures(const std::vector<double>& input, int degree, 
                                    int index, std::vector<int> powers, 
                                    std::vector<std::vector<int>>& allPowers);

    // Function to generate the polynomial features
    std::vector<std::vector<double>> createPolynomialFeatures(const std::vector<double>& input, int degree);

    // Split the line based on the delimiter
    std::vector<std::string> split(const std::string& line, char delimiter = ' ');
}

#endif