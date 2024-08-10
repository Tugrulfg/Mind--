#ifndef CSVREADER_HPP
#define CSVREADER_HPP

#include "Mind++.hpp"

#include <ostream>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <tuple>

namespace cmind{

    // Class for reading csv files and constructing the dataset
    class CSVReader{
        friend class Dataset;
        public:
            // CSV Dataset constructor
            CSVReader(const std::string& path, bool with_header=false);

            // Returns the column names
            const std::vector<std::string>& get_headers()const;

            // Returns the shape of the dataset
            Shape shape()const;

            // Returns the data at the given index
            std::tuple<const void*, dtype> at(const size_t row, const size_t col)const;

            // Sets the data at the given index
            void set(const size_t row, const size_t col, void* data);

            // Accessing a column with column name: r-value 
            std::tuple<const void*, dtype> operator[](const std::string& col)const;

            // Accessing a column with column name: l-value 
            std::tuple<void* , dtype> operator[](const std::string& col);

            // Accessing a column with index: r-value 
            std::tuple<void*, dtype> operator[](const size_t idx)const;

            // Accessing a column with index: l-value 
            std::tuple<void* , dtype> operator[](const size_t idx);
            
            // Destructor
            ~CSVReader();

            // Overloading the << operator
            friend std::ostream& operator<<(std::ostream& os, const CSVReader& dataset);
        
        private:
            // Reads the csv file and constructs the dataset
            void read(const std::string& path);

            // Splits the line into values according to ','
            static std::vector<std::string> split(const std::string& line);

            // Removes the empty spaces from the string
            static std::string trim(const std::string& str);

            const bool with_header;   // If the first row contains column names
            std::vector<std::string> headers;
            std::vector<void*> columns;
            size_t num_row = 0;
            size_t num_column = 0;
            std::vector<dtype> column_types;
    };
}






#endif