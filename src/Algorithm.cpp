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

     // Saves the weights
    void LinearRegression::save(const std::string& filepath)const{
        std::ofstream file(filepath);
        file << std::to_string((int)this->alg_type) + "\n";

        for(size_t i=0; i<this->weights_->shape()[0]; i++)
            file << std::to_string(this->weights_->data()[i]) + " ";
        
        file << "\nEOF";
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
        this->weights_ = new Tensor<float>({weights.size()});
        for(size_t i=0; i<weights.size(); i++)
            (*this->weights_)[i] = std::stof(weights[i]);
        
        file.close();
    }

    // Destructor
    LinearRegression::~LinearRegression(){
        delete this->weights_;
    }


    // PolynomialRegression constructor
    // PolynomialRegression::PolynomialRegression(const size_t input_count, const size_t degree): Algorithm(Algorithms::PolynomialRegression){
    //     this->num_var = new Tensor<float>({1});
    //     *this->num_var = input_count;
    //     this->degree = new Tensor<float>({1});
    //     *this->degree = degree;
    //     this->weights_ = new Tensor<float>({(factorial(*this->num_var+(*this->degree))/(factorial((*this->degree))*factorial(*this->num_var-(*this->degree))))});
    //     this->weights_->randomize();
    // }

    // // Load model from file
    // PolynomialRegression::PolynomialRegression(const std::string& filepath): Algorithm(Algorithms::PolynomialRegression){
    //     this->num_var = new Tensor<float>({1});
    //     this->degree = new Tensor<float>({1});
    //     this->load(filepath);
    // }

    // // Runs the algorithm and returns the result
    // const Tensor<float> PolynomialRegression::operator()(const Tensor<float>& input)const{
    //     Tensor<float> output({input.shape()[0]});
    //     output.fill(0.0);
    // }

    // // Trains the algorithm
    // void PolynomialRegression::train(Dataset& ds, const size_t epochs, Loss& loss_func, Optimizer& opt){

    // }

    // // Predicts for a single input
    // const Tensor<float> PolynomialRegression::predict(const std::vector<float> input)const{
    //     Tensor<float> output({1});
    //     output.fill(0.0);
    // }

    //  // Saves the weights
    // void PolynomialRegression::save(const std::string& filepath)const{
    //     std::ofstream file(filepath);
    //     file << std::to_string((int)this->alg_type) + "\n";
    //     file << std::to_string(this->num_var->data()[0]) + " " + std::to_string(this->degree->data()[0]) + "\n";

    //     for(size_t i=0; i<this->weights_->shape()[0]; i++)
    //         file << std::to_string(this->weights_->data()[i]) + " ";
        
    //     file << "\nEOF";
    // }

    // // Loads the weights
    // void PolynomialRegression::load(const std::string& filepath){
    //     std::ifstream file(filepath);
    //     std::string line;

    //     std::getline(file, line);
    //     if(stoi(line) != (int)this->alg_type){
    //         std::cout << "Load weights: Algorithm type does not match" << std::endl;
    //         abort();
    //     }
    //     std::getline(file, line);
    //     std::vector<std::string> num_var = split(line, ' ');
    //     *this->num_var = std::stoi(num_var[0]);
    //     *this->degree = std::stoi(num_var[1]);
    //     std::getline(file, line);
    //     std::vector<std::string> weights = split(line, ' ');
    //     this->weights_ = new Tensor<float>({weights.size()});
    //     for(size_t i=0; i<weights.size(); i++)
    //         (*this->weights_)[i] = std::stof(weights[i]);
        
    //     file.close();
    // }

    // // Destructor
    // PolynomialRegression::~PolynomialRegression(){
    //     delete this->num_var;
    //     delete this->degree;
    //     delete this->weights_;
    // }
}

