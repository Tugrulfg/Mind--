#include "../include/Optimizer.hpp"

namespace cmind{

    Optimizer::Optimizer(const Optimizers opt_type, const float lr): opt_type(opt_type), lr({1}){
        this->lr.fill(lr);
    }

    // Set the weights
    void Optimizer::set_weights(Tensor<float>* weights){
        this->weights = weights;
    }

    SGD::SGD(const float lr): Optimizer(Optimizers::SGD, lr){

    }

    void SGD::optimize(const Tensor<float>& grads)const{
        // std::cout << "Weights: " << this->weights->shape() << " Grads: " << grads.shape() << std::endl;
        for(size_t i=0; i<grads.shape()[0]; i++){
            (*this->weights)[i] -= this->lr*grads[i];
        }
    }
}