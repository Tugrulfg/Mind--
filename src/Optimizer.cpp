#include "../include/Optimizer.hpp"

namespace cmind{

    Optimizer::Optimizer(const Optimizers opt_type, const float lr): opt_type(opt_type), lr({1}){
        this->lr.fill(lr);
    }

    SGD::SGD(const float lr): Optimizer(Optimizers::SGD, lr){

    }

    void SGD::optimize(const Tensor<float>& grads)const{
        // std::cout << "Weights: " << this->weights->shape() << " Grads: " << grads.shape() << std::endl;
        for(size_t i=0; i<grads.shape()[0]; i++){
            (*this->weights)[i] -= this->lr*grads[i];
        }
    }

    // Set the weights
    void SGD::set_weights(Tensor<float>* weights){
        this->weights = weights;
    }

    SGDMomentum::SGDMomentum(const float lr, const float momentum): Optimizer(Optimizers::SGDMomentum, lr){
        this->momentum = new Tensor<float>({1});
        this->momentum->fill(momentum);
    }

    void SGDMomentum::optimize(const Tensor<float>& grads)const{
        // std::cout << "Optimize" << std::endl;
        // std::cout << "Weights: " << this->weights->shape() << " Grads: " << grads.shape() << " Velocity: " << this->velocity->shape() << std::endl;
        for(size_t i=0; i<grads.shape()[0]; i++){
            (*this->velocity)[i] = (*this->momentum)*(*this->velocity)[i] - this->lr*grads[i];
            (*this->weights)[i] += (*this->velocity)[i];
        }
    }

    void SGDMomentum::set_weights(Tensor<float>* weights){
        this->weights = weights;
        this->velocity = new Tensor<float>(this->weights->shape());
        this->velocity->fill(0.0);
    }

    SGDMomentum::~SGDMomentum(){
        if(this->momentum != nullptr)
            delete this->momentum;

        if(this->velocity != nullptr)
            delete this->velocity;
    }
}