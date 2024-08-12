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
        loss_func.set_weight_count(this->weights_->shape()[0]);
        opt.set_weights(this->weights_);
        size_t batch_count = ds.num_batches();
        Tensor<float>* data;
        Tensor<float>* target;
        Tensor<float> pred({ds.batch_size()});
        Tensor<float> loss({1});
        Tensor<float> grad({this->weights_->shape()[0]});

        std::cout << "Training" << std::endl;
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
    }

    // Destructor
    LinearRegression::~LinearRegression(){
        delete this->weights_;
    }
}