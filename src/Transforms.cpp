#include "../include/Transforms.hpp"

namespace cmind{

    Transforms::Transforms(const bool one_hot, const size_t poly_degree): one_hot(one_hot){
        if(poly_degree > 1){
            this->is_poly = true;

            this->poly_degree = new Tensor<size_t>({1});
            (*this->poly_degree)[0] = poly_degree;
        }
    }

    // Apply the transformations to the dataset
    void Transforms::fit_transform(Dataset& dataset){
        this->string_cols = dataset.string_cols;
        this->source_mapping = dataset.source_mapping;
        this->target_mapping = dataset.target_mapping;

        if(this->one_hot){                // One hot encoding
            std::vector<size_t> colls;
            size_t col_size = dataset.shape_[1]-this->string_cols.size();
            for(const std::unordered_map<std::string, float>& i: this->source_mapping){
                colls.push_back(i.size());
                col_size += colls.back();
            }

            Tensor<float>* output = nullptr;

            for(int i=0; i<dataset.num_batches(); i++){         // One hot encoding for data batches
                output = new Tensor<float>({dataset.batch_size_, col_size});
                output->fill(0.0);
                for(int k=0; k<dataset.batch_size_; k++){
                    size_t col = 0;
                    for(size_t j=0; j<dataset.shape_[1]; j++){
                        if(std::find(this->string_cols.begin(), this->string_cols.end(), j) == this->string_cols.end())
                            (*output)[k][col++] = (*dataset.data_batches[i])[k][j];
                        else{
                            (*output)[k][col+(*dataset.data_batches[i])[k][j].data()[0]] = 1.0;
                            col += colls[j];
                        }
                    }
                }

                delete dataset.data_batches[i];
                dataset.data_batches[i] = output;
            }
            dataset.shape_[1] = col_size;
        }

        if(this->is_poly)               // Generate polynomial features
            this->generate_combinations(dataset);
        
    }

    // Convert the test input to polynomial features
    std::vector<float> Transforms::operator()(const std::vector<float>& input)const{
        std::vector<float> output;
        size_t count = 0;

        if(this->one_hot){
            for(size_t i=0; i<input.size(); i++){
                if(std::find(this->string_cols.begin(), this->string_cols.end(), i) == this->string_cols.end())
                    output.push_back(input[i]);
                else{
                    for(size_t j=0; j<this->source_mapping[count].size(); j++){
                        if(input[i] == j)
                            output.push_back(1.0);
                        else
                            output.push_back(0.0);
                    }
                    count++;
                }
            }
        }
        else
            output = input;
        

        if (this->is_poly) {
            std::vector<std::vector<float>> polyFeatures = createPolynomialFeatures(output, this->poly_degree->data()[0]);
            output.clear();
            for (const auto& feature : polyFeatures) 
                output.push_back(feature[0]);
        }
            
        return output;
    }

    // Generate all polynomial combinations
    void Transforms::generate_combinations(Dataset& dataset)const{
        size_t n = dataset.shape_[1]; // Number of features
        size_t m = dataset.shape_[0]; // Number of samples

        int degree_val = this->poly_degree->data()[0];
        size_t num_output_features = 1;
        
        Tensor<float>* output;
        std::vector<std::vector<std::vector<float>>> batch;
        std::vector<float> example;
            for (int i = 0; i < dataset.num_batches(); ++i) {                
                for(int k=0; k<dataset.batch_size_; k++){
                    for(int j=0; j<n; j++)
                        example.push_back((*dataset.data_batches[i])[k][j].data()[0]);
                    
                    batch.push_back(this->createPolynomialFeatures(example, degree_val));
                    if(num_output_features == 1){
                        num_output_features = batch[0].size();

                    }
                    example.clear();
                }
    
                output = new Tensor<float>({dataset.batch_size_, num_output_features});
                for(int k=0; k<dataset.batch_size_; k++){
                    for(int j=0; j<num_output_features; j++)
                        (*output)[k][j] = batch[k][j][0];
                }
                delete dataset.data_batches[i];
                dataset.data_batches[i] = output;
                batch.clear();
            }
        
        dataset.shape_[1] = num_output_features;
    }

    // Helper function to recursively generate polynomial features
    void Transforms::generatePolynomialFeatures(const std::vector<float>& input, int degree, 
                                    int index, std::vector<int> powers, 
                                    std::vector<std::vector<int>>& allPowers) {
        if (index == input.size()) {
            int sum = 0;
            for (int power : powers) {
                sum += power;
            }
            if (sum > 0 && sum <= degree) {  // Skip the bias term (sum > 0)
                allPowers.push_back(powers);
            }
            return;
        }

        for (int i = 0; i <= degree; ++i) {
            powers[index] = i;
            generatePolynomialFeatures(input, degree, index + 1, powers, allPowers);
        }
    }

    std::vector<std::vector<float>> Transforms::createPolynomialFeatures(const std::vector<float>& input, int degree) {
        std::vector<std::vector<int>> allPowers;
        std::vector<int> powers(input.size(), 0);
        
        // Generate all combinations of powers for each feature
        generatePolynomialFeatures(input, degree, 0, powers, allPowers);

        // Now convert these powers into actual polynomial features
        std::vector<std::vector<float>> polyFeatures;
        for (const auto& powerSet : allPowers) {
            float product = 1.0;
            for (int i = 0; i < input.size(); ++i) {
                product *= std::pow(input[i], powerSet[i]);
            }
            polyFeatures.push_back({product});
        }
        
        return polyFeatures;
    }


    Transforms::~Transforms(){
        if(this->poly_degree != nullptr)
            delete this->poly_degree;

    }
}