#include "../include/Dataset.hpp"
#include "../include/Math.hpp"

namespace cmind{
    // Dataset constructor
    Dataset::Dataset(const CSVReader& csv, const size_t target, const std::vector<size_t>& ignore, size_t batch, bool shuffle): shape_({csv.shape()[0], csv.shape()[1] - ignore.size() - 1}), batch_size_(batch){
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

        this->data_batches = std::vector<Tensor<float>*>(this->num_batches());
        this->target_batches = std::vector<Tensor<float>*>(this->num_batches());
        for(size_t i=0; i<this->num_batches(); i++){
            this->data_batches[i] = new Tensor<float>({this->batch_size_, this->shape_[1]});
            this->target_batches[i] = new Tensor<float>({this->batch_size_});
        }

        Tensor<float>* source_data = new Tensor<float>({this->shape_[1], this->shape_[0]});        
        Tensor<float>* target_data;

        bool found;
        if(!shuffle){
            size_t row = 0;
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
                        target_data = new Tensor<float>(to_float(csv.columns[i], dtype::INT, row_count));
                    else if(csv.column_types[i] == dtype::FLOAT)
                        target_data = new Tensor<float>(to_float(csv.columns[i], dtype::FLOAT, row_count));
                    else if(csv.column_types[i] == dtype::BOOL)
                        target_data = new Tensor<float>(to_float(csv.columns[i], dtype::BOOL, row_count));
                    else if(csv.column_types[i] == dtype::STR){
                        this->target_mapping = string_to_float(csv.columns[i], row_count);
                        target_data = new Tensor<float>({row_count});

                        for(size_t j=0; j<row_count; j++)
                            (*target_data)[j] = this->target_mapping[*(*(Tensor<std::string>*)std::get<0>(csv[i]))[j].data()];
                    }
                }
                else if(csv.column_types[i] == dtype::INT)
                    (*source_data)[row++] = to_float(csv.columns[i], dtype::INT, row_count);
                else if(csv.column_types[i] == dtype::FLOAT)
                    (*source_data)[row++] = to_float(csv.columns[i], dtype::FLOAT, row_count);
                else if(csv.column_types[i] == dtype::BOOL)
                    (*source_data)[row++] = to_float(csv.columns[i], dtype::BOOL, row_count);
                else if(csv.column_types[i] == dtype::STR){
                    std::unordered_map<std::string, float> string_mapping = string_to_float(csv.columns[i], row_count);
                    Tensor<float> tensor({row_count});
                    for(size_t j=0; j<row_count; j++)
                        tensor[j] = string_mapping[*(*(Tensor<std::string>*)std::get<0>(csv[i]))[j].data()];
                    (*source_data)[row++] = tensor;
                }
            }
        }

        for(size_t i=0; i<this->num_batches(); i++){
            for(size_t j=0; j<this->batch_size_; j++){
                (*this->target_batches[i])[j] = (*target_data)[i*this->batch_size_ + j];
                for(size_t k=0; k<this->shape_[1]; k++)
                    (*this->data_batches[i])[j][k] = (*source_data)[k][i*this->batch_size_ + j];
            }
        }

        delete source_data;
        delete target_data;
    }

    // Return the shape of the dataset
    Shape Dataset::shape() const{
        return this->shape_;
    }

    // Gets the next batch of data
    Tensor<float>* Dataset::next_data(){
        return this->data_batches[this->data_batch_index++];
    }

    // Gets the next batch of targets
    Tensor<float>* Dataset::next_targets(){
        return this->target_batches[this->target_batch_index++];
    }

    // Number of batches
    size_t Dataset::num_batches()const{
        return this->shape_[0] / this->batch_size_;
    }

    // Returns the batch size
    size_t Dataset::batch_size()const{
        return this->batch_size_;
    }

    // Sets the indices to beginning
    void Dataset::reset(){
        this->data_batch_index = 0;
        this->target_batch_index = 0;
    }

    // Preprocessing
    void Dataset::preprocess(const size_t poly_feature_degree){
        if(poly_feature_degree > 1){
            Tensor<size_t> degree_tensor({1});
            degree_tensor[0] = poly_feature_degree;
            this->generate_combinations(degree_tensor);
        }
    }

    // Destructor
    Dataset::~Dataset(){
        for(auto data : this->data_batches)
            delete data;
        for(auto data : this->target_batches)
            delete data;
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

    // Generate all polynomial combinations
    void Dataset::generate_combinations(const Tensor<size_t>& degree){
        size_t n = this->shape_[1]; // Number of features
        size_t m = this->shape_[0]; // Number of samples

        int degree_val = degree.data()[0];
        size_t num_output_features = 1;
        
        Tensor<float>* output;
        std::vector<std::vector<std::vector<double>>> batch;
        std::vector<double> example;
            for (int i = 0; i < this->num_batches(); ++i) {                
                for(int k=0; k<batch_size_; k++){
                    for(int j=0; j<n; j++)
                        example.push_back((*this->data_batches[i])[k][j].data()[0]);
                    
                    batch.push_back(createPolynomialFeatures(example, degree_val));
                    if(num_output_features == 1){
                        num_output_features = batch[0].size();

                    }
                    example.clear();
                }
    
                output = new Tensor<float>({this->batch_size_, num_output_features});
                for(int k=0; k<batch_size_; k++){
                    for(int j=0; j<num_output_features; j++)
                        (*output)[k][j] = batch[k][j][0];
                }
                delete this->data_batches[i];
                this->data_batches[i] = output;
                batch.clear();
            }
        
        this->shape_[1] = num_output_features;
    }

    // Overloading the << operator
    std::ostream& operator<<(std::ostream& os, const Dataset& dataset){
        os << std::endl << "------------------------------------------------------------------------------------" << std::endl;

        for(size_t i=0; i<dataset.headers.size(); i++)
            os << dataset.headers[i] << "\t";
        
        os << "|\t" << dataset.target_header;
        os << std::endl << "------------------------------------------------------------------------------------" << std::endl;
        for(size_t i=0; i<dataset.num_batches(); i++){
            for(size_t j=0; j<dataset.batch_size_; j++){
                for(size_t k=0; k<dataset.shape_[1]; k++)
                    os << (*dataset.data_batches[i])[j][k] << "\t";
                os << "|\t" << (*dataset.target_batches[i])[j];
                os << std::endl;
            }
        }
        os << "------------------------------------------------------------------------------------" << std::endl << std::endl;
        return os;
    }
}