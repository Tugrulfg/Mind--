#ifndef TRANSFORMS_HPP
#define TRANSFORMS_HPP

#include "Mind++.hpp"
#include <unordered_map>

namespace cmind{

    // Class to store and perform the transformations applied on the dataset before feeding to model
    class Transforms{
        public:
            Transforms(const bool one_hot = false,const size_t poly_degree = 1);

            // Apply the transformations to the dataset
            void fit_transform(Dataset& dataset);

            // Convert the test input to polynomial features
            std::vector<float> operator()(const std::vector<float>& input)const;

            // Destructor
            ~Transforms();

        private:
            // Generate all polynomial combinations
            void generate_combinations(Dataset& dataset)const;
            
            // Helper function to recursively generate polynomial features
            static void generatePolynomialFeatures(const std::vector<float>& input, int degree, 
                                            int index, std::vector<int> powers, 
                                            std::vector<std::vector<int>>& allPowers);

            // Function to generate the polynomial features
            static std::vector<std::vector<float>> createPolynomialFeatures(const std::vector<float>& input, int degree);
            
            bool is_poly = false;
            bool one_hot = false;
            Tensor<size_t>* poly_degree = nullptr;
            std::vector<size_t> string_cols;
            std::vector<std::unordered_map<std::string, float>> source_mapping; // If the source column is string, store the mapping
            std::unordered_map<std::string, float> target_mapping;     // If the target column is string, store the mapping
    };


}

#endif