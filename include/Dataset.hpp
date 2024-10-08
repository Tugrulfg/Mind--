#ifndef DATASET_HPP
#define DATASET_HPP

#include "Mind++.hpp"

#include <algorithm>  // For std::shuffle
#include <random>     // For std::default_random_engine
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <limits>
#include <ostream>

namespace cmind{
    class CSVReader;  // Forward declare CSVReader

    class Dataset{
        friend class Transforms;
        public:
            // Dataset constructor
            Dataset(const CSVReader& csv, const size_t target = std::numeric_limits<size_t>::max(), const std::vector<size_t>& ignore = {}, size_t batch = 1, bool shuffle_data = false);

            // Returns the shape of the dataset
            Shape shape()const;

            // Gets the next batch of data
            Tensor<float>* next_data();

            // Gets the next batch of targets
            Tensor<float>* next_targets();

            // Number of batches
            size_t num_batches()const;

            // Returns the batch size
            size_t batch_size()const;

            // Sets the indices to beginning
            void reset();

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
            static Tensor<float> to_float(const void* data, dtype type, size_t size);

            std::vector<Tensor<float>*> data_batches;
            std::vector<Tensor<float>*> target_batches;
            Shape shape_;
            const size_t batch_size_;
            std::vector<size_t> string_cols;
            std::vector<std::unordered_map<std::string, float>> source_mapping; // If the source column is string, store the mapping
            std::unordered_map<std::string, float> target_mapping;     // If the target column is string, store the mapping
            size_t data_batch_index = 0;   // Current row index for the dataset
            size_t target_batch_index = 0; // Current row index for the target
    };
}

#endif