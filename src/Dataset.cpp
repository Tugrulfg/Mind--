#include "../include/Dataset.hpp"

namespace cmind{
    // Dataset constructor
    Dataset::Dataset(const CSVReader& csv, const size_t target, const std::vector<size_t>& ignore, size_t batch, bool shuffle): shape_({csv.shape()[0], csv.shape()[1] - ignore.size() - 1}), batch_size(batch){
        if(csv.shape()[1] - ignore.size() <= 0){
            std::cout << "Dataset construction: Empty dataset" << std::endl;
            abort();
        }
        else if(is_duplicate(ignore)){
            std::cout << "Dataset construction: Duplicated ignored indices" << std::endl;
            abort();
        }
        else{
            for(size_t idx: ignore){
                if(idx >= csv.shape()[1]){
                    std::cout << "Dataset construction: Ignored indices out of range" << std::endl;
                    abort();
                }
            }
        }

        bool found;
        if(!shuffle){
            for(size_t i=0; i<csv.shape()[1]; i++){
                found = false;
                for(size_t idx: ignore){       // Check if the column is ignored or not
                    if(i == idx){
                        found = true;
                        break;
                    }
                }
                if(found)           
                    continue;
                
                if(i != target)
                    this->headers.push_back(csv.headers[i]);
                
                size_t row_count = csv.shape()[0];

                if(i == target || (target == std::numeric_limits<size_t>::max() && i == this->shape_[1]-1)){                // Target column
                    this->target_header = csv.headers[i];
                    if(csv.column_types[i] == dtype::INT)
                        this->target_data = new Tensor<float>(to_float(csv.columns[i], dtype::INT, row_count));
                    else if(csv.column_types[i] == dtype::FLOAT)
                        this->target_data = new Tensor<float>(to_float(csv.columns[i], dtype::FLOAT, row_count));
                    else if(csv.column_types[i] == dtype::BOOL)
                        this->target_data = new Tensor<float>(to_float(csv.columns[i], dtype::BOOL, row_count));
                    else if(csv.column_types[i] == dtype::STR){
                        this->target_mapping = string_to_float(csv.columns[i], row_count);
                        this->target_data = new Tensor<float>({row_count});

                        for(size_t j=0; j<row_count; j++)
                            (*this->target_data)[j] = this->target_mapping[*(*(Tensor<std::string>*)std::get<0>(csv[5]))[j].data()];
                    }
                }
                else if(csv.column_types[i] == dtype::INT)
                    this->source_data.push_back(new Tensor<float>(to_float(csv.columns[i], dtype::INT, row_count)));
                else if(csv.column_types[i] == dtype::FLOAT)
                    this->source_data.push_back(new Tensor<float>(to_float(csv.columns[i], dtype::FLOAT, row_count)));
                else if(csv.column_types[i] == dtype::BOOL)
                    this->source_data.push_back(new Tensor<float>(to_float(csv.columns[i], dtype::BOOL, row_count)));
                else if(csv.column_types[i] == dtype::STR){
                    std::unordered_map<std::string, float> string_mapping = string_to_float(csv.columns[i], row_count);
                    Tensor<float>* tensor = new Tensor<float>({row_count});
                    for(size_t j=0; j<row_count; j++)
                        tensor[j] = string_mapping[*(*(Tensor<std::string>*)std::get<0>(csv[5]))[j].data()];
                    this->source_data.push_back(tensor);
                }

            }
        }
    }

    // Return the shape of the dataset
    Shape Dataset::shape() const{
        return this->shape_;
    }

    // Returns the data at the given index
    float Dataset::at(const size_t row, const size_t col) const{
        if(row >= this->shape_[0] || col >= this->shape_[1]){
            std::cerr << "Dataset: Index out of bounds: " << row << " " << col << std::endl;
            abort();
        }
        return (float)(*(this->source_data[col])[row].data());
    }

    // Returns the target data
    float Dataset::target_at(const size_t row) const{
        if(row >= this->shape_[0]){
            std::cerr << "Dataset: Index out of bounds: " << row << std::endl;
            abort();
        }
        return *(float*)(*target_data)[row].data();
    }

    // Returns the selected row
    Tensor<float> Dataset::row(const size_t row) const{
        if(row >= this->shape_[0]){
            std::cerr << "Dataset: Index out of bounds: " << row << std::endl;
            abort();
        }
        float* row_data = new float[this->shape_[1]];
        for(size_t i=0; i<this->shape_[1]; i++)
            row_data[i] = this->at(row, i);
        return Tensor<float>(row_data, {this->shape_[1]});
    }

    // Returns the selected row: Returned array needs to be deallocated by the user
    const Tensor<float>& Dataset::col(const size_t col) const{
        if(col >= this->shape_[1]){
            std::cerr << "Dataset: Index out of bounds: " << col << std::endl;
            abort();
        }
        return *this->source_data[col];
    }

    // Returns the targets
    const Tensor<float>& Dataset::targets() const{
        return *this->target_data;
    }

    // Destructor
    Dataset::~Dataset(){
        for(size_t i=0; i<this->shape_[1]; i++)
            delete this->source_data[i];
        
        delete this->target_data;
    }

    // Check if the ignored indices have duplicate
    bool Dataset::is_duplicate(const std::vector<size_t>& ignore){
        std::unordered_set<size_t> seen;
        for (size_t num : ignore) {
            if (seen.count(num) > 0) {
                return true; // Found a duplicate
            }
            seen.insert(num);
        }
        return false; // No duplicates found
    }

    // Creates mapping for converting string data to numeric data
    std::unordered_map<std::string, float> Dataset::string_to_float(const void* data, const size_t size){
        std::unordered_map<std::string, float> mapping;
        Tensor<std::string> tensor = *(Tensor<std::string>*)data;
        int pos = 0;
        for(size_t i=0; i<size; i++){
            if(mapping.find(*((std::string*)tensor[i].data())) == mapping.end())       // If the value is not in the mapping, add a new value
                mapping[*((std::string*)tensor[i].data())] = pos++;
        }
        return mapping;
    }

    // Convert tensor to float
    Tensor<float> Dataset::to_float(void* data, dtype type, size_t size){
        Tensor<float> tensor({size});
        if(type == dtype::INT){
            for(size_t i=0; i<size; i++)
                tensor[i] = (float)((*(Tensor<int>*)data).data()[i]);
        }
        else if(type == dtype::BOOL){
            for(size_t i=0; i<size; i++)
                tensor[i] = (float)((*(Tensor<bool>*)data).data()[i]);
        }
        else if(type == dtype::FLOAT){
            for(size_t i=0; i<size; i++)
                tensor[i] = (float)((*(Tensor<float>*)data).data()[i]);
        }
        return tensor;
    }

    // Overloading the << operator
    std::ostream& operator<<(std::ostream& os, const Dataset& dataset){
        os << std::endl << "------------------------------------------------------------------------------------" << std::endl;

        for(size_t i=0; i<dataset.shape_[1]; i++)
            os << dataset.headers[i] << "\t";
        
        os << "|\t" << dataset.target_header;
        os << std::endl << "------------------------------------------------------------------------------------" << std::endl;
        for(size_t i=0; i<dataset.shape_[0]; i++){
            for(size_t j=0; j<dataset.shape_[1]; j++){
                os << (*dataset.source_data[j])[i] << "\t";
            }
            os << "|\t";
            os << (*dataset.target_data)[i];

            os << std::endl;
        }
        os << "------------------------------------------------------------------------------------" << std::endl << std::endl;
        return os;
    }
}