#include "../include/Algorithm.hpp"

namespace cmind{
    // Constructor
    Algorithm::Algorithm(const Algorithms alg_type): alg_type(alg_type){
        
    }

    // Returns the weights
    const Tensor<float>* Algorithm::weights()const{
        return this->weights_;
    }

    // Saves the weights
    void Algorithm::save(const std::string& filepath)const{
        std::ofstream file(filepath);
        file << std::to_string((int)this->alg_type) + "\n";

        for(size_t i=0; i<this->weights_->shape()[0]; i++)
            file << std::to_string(this->weights_->data()[i]) + " ";
        
        file << "\nEOF";
    }

    // Loads the weights
    void Algorithm::load(const std::string& filepath){
        std::ifstream file(filepath);
        std::string line;

        std::getline(file, line);
        if(stoi(line) != (int)this->alg_type){
            std::cout << "Load weights: Algorithm type does not match" << std::endl;
            abort();
        }
        std::getline(file, line);
        std::vector<std::string> weights = split(line);
        this->weights_ = new Tensor<float>({weights.size()});
        for(size_t i=0; i<weights.size(); i++)
            (*this->weights_)[i] = std::stof(weights[i]);
        
        file.close();
    }

    // Splits the line into values according to ' '
    std::vector<std::string> Algorithm::split(const std::string& line){
        std::vector<std::string> values;
        std::stringstream ss(line);
        std::string token;

        while (std::getline(ss, token, ' ')) {
            token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());    // Trim spaces
            if(!token.empty())
                values.push_back(token);
        }

        return values;
    }

    // LinearRegression constructor
    LinearRegression::LinearRegression(const size_t input_count): Algorithm(Algorithms::LinearRegression){
        this->weights_ = new Tensor<float>({input_count+1});
        this->weights_->randomize();
    }

    // Load model from file
    LinearRegression::LinearRegression(const std::string& filepath): Algorithm(Algorithms::LinearRegression){
        std::ifstream file(filepath);

        this->load(filepath);
    }

    // Runs the algorithm and returns the result
    const Tensor<float> LinearRegression::operator()(const Tensor<float>& input)const{
        Tensor<float> output({input.shape()[0]});
        output.fill(0.0);

        for(size_t i=0; i<input.shape()[0]; i++){
            for(size_t j=0; j<this->weights_->shape()[0]-1; j++)
                output[i] += (*this->weights_)[j]*input[i][j];
            output[i] += (*this->weights_)[this->weights_->shape()[0]-1];
        }
        return output;
    }

    // Trains the algorithm
    void LinearRegression::train(Dataset& ds, const size_t epochs, Loss& loss_func, Optimizer& opt){
        loss_func.set_alg_type(this->alg_type);
        loss_func.set_weight_count(this->weights_->shape()[0]);
        opt.set_weights(this->weights_);
        size_t batch_count = ds.num_batches();
        Tensor<float>* data;
        Tensor<float>* target;
        Tensor<float> pred({ds.batch_size()});
        Tensor<float> loss({1});
        Tensor<float> grad({this->weights_->shape()[0]});

        std::cout << "Training" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for(size_t i=0; i<epochs; i++){
            for(size_t j=0; j<batch_count; j++){
                data = ds.next_data();
                target = ds.next_targets();
                pred = this->operator()(*data);

                loss_func.set_inputs(data);

                loss = loss_func.compute(pred, *target);

                grad = loss_func.gradient(pred, *target);
                opt.optimize(grad);
                // std::cout << "\rEpoch: " << i+1 << ", Batch: " << j+1 << " Preds: " << pred << " Targets: " << *target << " Loss: " << loss << std::flush;
                std::cout << "\rEpoch: " << i+1 << ", Batch: " << j+1 << " Loss: " << loss << std::flush;
            }
            std::cout << std::endl;
            ds.reset();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        std::cout << "Training time: " << time/1000 << "." << time%1000 << "s" << std::endl << std::endl;
    }

    // Predicts for a single input
    const Tensor<float> LinearRegression::predict(const std::vector<float> input)const{
        Tensor<float> output({1});
        output.fill(0.0);

        for(size_t j=0; j<this->weights_->shape()[0]-1; j++)
            output += (*this->weights_)[j]*input[j];
        output += (*this->weights_)[this->weights_->shape()[0]-1];
        
        return output;
    }

    // Destructor
    LinearRegression::~LinearRegression(){
        delete this->weights_;
    }
}