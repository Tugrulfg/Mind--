#include "../include/Algorithm.hpp"

namespace cmind{
    // Constructor
    Algorithm::Algorithm(const Algorithms alg_type): alg_type(alg_type){
        
    }

    // Returns the weights
    const Tensor<float>* Algorithm::weights()const{
        return this->weights_;
    }

    // LinearRegression constructor
    LinearRegression::LinearRegression(const size_t input_count): Algorithm(Algorithms::LinearRegression){
        this->weights_ = new Tensor<float>({input_count,1});
        this->weights_->randomize();

        this->bias_ = new Tensor<float>({1});
        this->bias_->randomize(); 
    }

    // Load model from file
    LinearRegression::LinearRegression(const std::string& filepath): Algorithm(Algorithms::LinearRegression){
        std::ifstream file(filepath);

        this->load(filepath);
    }

    // Returns the bias
    const Tensor<float>* LinearRegression::bias()const{
        return this->bias_;
    }

    // Runs the algorithm and returns the result
    const Tensor<float> LinearRegression::operator()(const Tensor<float>& input)const{
        // Tensor<float> output({input.shape()[0]});
        // output.fill(0.0);

        // for(size_t i=0; i<input.shape()[0]; i++){
        //     for(size_t j=0; j<this->weights_->shape()[0]; j++)
        //         output[i] += (*this->weights_)[j]*input[i][j];
        //     output[i] += (*this->bias_)[0];
        // }
        // return output;
        Tensor<float> output({input.shape()[0], 1});
        output.fill(this->bias_->data()[0]);

        output += mat_mul(input, *this->weights_);
        return output.flatten();
    }

    // Sets the loss function and the optimizer
    void LinearRegression::compile(Loss& loss_func, Optimizer& opt){
        this->loss_func_ = &loss_func;
        this->opt_ = &opt;
    }

    // Trains the algorithm
    void LinearRegression::train(Dataset& ds, const size_t epochs){
        if(this->loss_func_->get_loss_type() == Losses::RidgeLoss)
            ((RidgeLoss*)this->loss_func_)->set_weights(this->weights_, this->bias_);
        else if(this->loss_func_->get_loss_type() == Losses::LassoLoss)
            ((LassoLoss*)this->loss_func_)->set_weights(this->weights_, this->bias_);
        else if(this->loss_func_->get_loss_type() == Losses::ElasticNetLoss)
            ((ElasticNetLoss*)this->loss_func_)->set_weights(this->weights_, this->bias_);

        this->loss_func_->set_alg_type(this->alg_type);
        this->loss_func_->set_weight_count(this->weights_->shape()[0]);
        this->opt_->set_weights(this->weights_, this->bias_);
        size_t batch_count = ds.num_batches();
        Tensor<float>* data;
        Tensor<float>* target;
        Tensor<float> pred({ds.batch_size()});
        Tensor<float> loss({1});
        Tensor<float> grad({this->weights_->shape()[0]+1});

        std::cout << "Training" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for(size_t i=0; i<epochs; i++){
            for(size_t j=0; j<batch_count; j++){
                data = ds.next_data();
                target = ds.next_targets();

                pred = this->operator()(*data);

                this->loss_func_->set_inputs(data);

                loss = this->loss_func_->compute(pred, *target);

                grad = this->loss_func_->gradient(pred, *target);
                this->opt_->optimize(grad);
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
        if(input.size() != this->weights_->shape()[0]){
            std::cout << "Predict: Input size does not match" << std::endl;
            abort();
        }
        Tensor<float> output({1});
        output.fill(0.0);

        for(size_t j=0; j<this->weights_->shape()[0]; j++)
            output += (*this->weights_)[j]*input[j];
        output += (*this->bias_)[0];
        
        return output;
    }

     // Saves the weights
    void LinearRegression::save(const std::string& filepath)const{
        std::ofstream file(filepath);
        file << std::to_string((int)this->alg_type) + "\n";

        for(size_t i=0; i<this->weights_->shape()[0]; i++)
            file << std::to_string(this->weights_->data()[i]) + " ";
        file << std::to_string(this->bias_->data()[0]) + " \n";
        file << "EOF";
    }

    // Loads the weights
    void LinearRegression::load(const std::string& filepath){
        std::ifstream file(filepath);
        std::string line;

        std::getline(file, line);
        if(stoi(line) != (int)this->alg_type){
            std::cout << "Load weights: Algorithm type does not match" << std::endl;
            abort();
        }
        std::getline(file, line);
        std::vector<std::string> weights = split(line, ' ');

        if(this->weights_ == nullptr)
            delete this->weights_;
        this->weights_ = new Tensor<float>({weights.size()-1});
        for(size_t i=0; i<weights.size()-1; i++)
            (*this->weights_)[i] = std::stof(weights[i]);
        
        if(this->bias_ != nullptr)
            this->bias_ = new Tensor<float>({1});
        (*this->bias_)[0] = std::stof(weights[weights.size()-1]);
        
        file.close();
    }

    // Destructor
    LinearRegression::~LinearRegression(){
        delete this->weights_;
        delete this->bias_;
    }
}

