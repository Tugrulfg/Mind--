#ifndef DATASET_HPP
#define DATASET_HPP

#include "Mind++.hpp"

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <limits>
#include <ostream>

namespace cmind{
    class CSVReader;  // Forward declare CSVReader

    class Dataset{
        public:
            // Dataset constructor
            Dataset(const CSVReader& csv, const size_t target = std::numeric_limits<size_t>::max(), const std::vector<size_t>& ignore = {}, size_t batch = 1, bool shuffle = false);

            // Preprocessing
            void preprocess();

            // Returns the shape of the dataset
            Shape shape()const;

            // Returns the data at the given index
            float at(const size_t row, const size_t col)const;

            // Returns the target data at the given index
            float target_at(const size_t row)const;

            // Returns the selected row: Returned array needs to be deallocated by the user
            Tensor<float> row(const size_t row)const;

            // Returns the selected column
            const Tensor<float>& col(const size_t col)const;

            // Returns the targets
            const Tensor<float>& targets()const;

            // Gets the next batch of data
            std::vector<Tensor<float>> next_data();

            // Gets the next batch of targets
            std::vector<Tensor<float>> next_targets();

            // Destructor
            ~Dataset();

             // Overloading the << operator
            friend std::ostream& operator<<(std::ostream& os, const Dataset& dataset);

        private:
            // Check if the ignored indices have duplicate
            static bool is_duplicate(const std::vector<size_t>& ignore);

            // Create mapping for converting string data to numeric data
            static std::unordered_map<std::string, float> string_to_float(const void* data, const size_t size);

            // Convert tensor to float
            static Tensor<float> to_float(void* data, dtype type, size_t size);

            std::vector<std::string> headers;
            std::vector<Tensor<float>*> source_data;
            Tensor<float>* target_data;
            std::string target_header;
            const Shape shape_;
            const size_t batch_size;
            std::unordered_map<std::string, float> target_mapping;     // If the target column is string, store the mapping
            size_t data_row_index = 0;   // Current row index for the dataset
            size_t target_row_index = 0; // Current row index for the target
    };
}

#endif