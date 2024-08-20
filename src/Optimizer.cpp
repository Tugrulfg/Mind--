#include "../include/Optimizer.hpp"

namespace cmind{

    Optimizer::Optimizer(const Optimizers opt_type, const float lr): opt_type(opt_type), lr({1}){
        this->lr.fill(lr);
    }

    SGD::SGD(const float lr): Optimizer(Optimizers::SGD, lr){

    }

    void SGD::optimize(const Tensor<float>& grads)const{
        // std::cout << "Weights: " << this->weights->shape() << " Grads: " << grads.shape() << std::endl;
        size_t count = this->weights->shape()[0];
        for(size_t i=0; i<count; i++)
            (*this->weights)[i] -= this->lr*grads[i];
        
        if(this->bias != nullptr)
            (*this->bias) -= this->lr*grads[count];
    }

    // Set the weights
    void SGD::set_weights(Tensor<float>* weights, Tensor<float>* bias){
        this->weights = weights;
        this->bias = bias;
    }

    SGDMomentum::SGDMomentum(const float lr, const float momentum): Optimizer(Optimizers::SGDMomentum, lr){
        this->momentum = new Tensor<float>({1});
        this->momentum->fill(momentum);
    }

    void SGDMomentum::optimize(const Tensor<float>& grads)const{
        // std::cout << "Optimize" << std::endl;
        // std::cout << "Weights: " << this->weights->shape() << " Grads: " << grads.shape() << " v: " << this->v->shape() << std::endl;
        size_t count = this->weights->shape()[0];
        for(size_t i=0; i<count; i++){
            (*this->v)[i] = (*this->momentum)*(*this->v)[i] - this->lr*grads[i];
            (*this->weights)[i] += (*this->v)[i];
        }
        if(this->bias != nullptr){
            (*this->v)[count] = (*this->momentum)*(*this->v)[count] - this->lr*grads[count];
            (*this->bias) += (*this->v)[count];   
        }
    }

    void SGDMomentum::set_weights(Tensor<float>* weights, Tensor<float>* bias){
        this->weights = weights;
        this->bias = bias;
        if(this->bias == nullptr)
            this->v = new Tensor<float>({this->weights->shape()[0]});
        else
            this->v = new Tensor<float>({this->weights->shape()[0]+1});
        this->v->fill(0.0);
    }

    SGDMomentum::~SGDMomentum(){
        if(this->momentum != nullptr)
            delete this->momentum;

        if(this->v != nullptr)
            delete this->v;
    }

    AdaGrad::AdaGrad(const float lr): Optimizer(Optimizers::AdaGrad, lr){

    }

    void AdaGrad::set_weights(Tensor<float>* weights, Tensor<float>* bias){
        this->weights = weights;
        this->bias = bias;
        if(this->bias == nullptr)
            this->v = new Tensor<float>({this->weights->shape()[0]});
        else
            this->v = new Tensor<float>({this->weights->shape()[0]+1});
        this->v->fill(0.0);
    }

    void AdaGrad::optimize(const Tensor<float>& grads)const{
        // std::cout << "Optimize" << std::endl;
        // std::cout << "Weights: " << this->weights->shape() << " Grads: " << grads.shape() << " v: " << this->v->shape() << std::endl;
        size_t count = this->weights->shape()[0];
        for(size_t i=0; i<count; i++){
            (*this->v)[i] += grads[i]*grads[i];
            (*this->weights)[i] -= this->lr*grads[i]/(sqrt((*this->v)[i])+1e-8);
        }
        if(this->bias != nullptr){
            (*this->v)[count] += grads[count]*grads[count];
            (*this->bias) -= this->lr*grads[count]/(sqrt((*this->v)[count])+1e-8);
        }
    }

    AdaGrad::~AdaGrad(){
        if(this->v != nullptr)
            delete this->v;
    }


    RMSProp::RMSProp(const float lr, const float beta): Optimizer(Optimizers::RMSProp, lr){
        this->beta = new Tensor<float>({1});
        this->beta->fill(beta);
    }

    void RMSProp::set_weights(Tensor<float>* weights, Tensor<float>* bias){
        this->weights = weights;
        this->bias = bias;
        if(this->bias == nullptr)
            this->v = new Tensor<float>({this->weights->shape()[0]});
        else
            this->v = new Tensor<float>({this->weights->shape()[0]+1});
        this->v->fill(0.0);
    }

    void RMSProp::optimize(const Tensor<float>& grads)const{
        // std::cout << "Optimize" << std::endl;
        // std::cout << "Weights: " << this->weights->shape() << " Grads: " << grads.shape() << " v: " << this->v->shape() << std::endl;
        size_t count = this->weights->shape()[0];
        for(size_t i=0; i<count; i++){
            (*this->v)[i] = (*this->beta)*(*this->v)[i] + ((float)(1.0)-(*this->beta))*grads[i]*grads[i];
            (*this->weights)[i] -= this->lr*grads[i]/(sqrt((*this->v)[i])+1e-8);
        }
        if(this->bias != nullptr){
            (*this->v)[count] = (*this->beta)*(*this->v)[count] + ((float)(1.0)-(*this->beta))*grads[count]*grads[count];
            (*this->bias) -= this->lr*grads[count]/(sqrt((*this->v)[count])+1e-8);
        }
    }

    RMSProp::~RMSProp(){
        if(this->beta != nullptr)
            delete this->beta;
        if(this->v != nullptr)
            delete this->v;
    }


    Adam::Adam(const float lr, const float beta1, const float beta2): Optimizer(Optimizers::Adam, lr){
        this->beta1 = new Tensor<float>({1});
        this->beta1->fill(beta1);
        this->beta2 = new Tensor<float>({1});
        this->beta2->fill(beta2);
    }

    void Adam::set_weights(Tensor<float>* weights, Tensor<float>* bias){
        this->weights = weights;
        this->bias = bias;
        if(this->bias == nullptr){
            this->m = new Tensor<float>({this->weights->shape()[0]});
            this->v = new Tensor<float>({this->weights->shape()[0]});
        }
        else{
            this->m = new Tensor<float>({this->weights->shape()[0]+1});
            this->v = new Tensor<float>({this->weights->shape()[0]+1});
        }
            
        this->m->fill(0.0);
        this->v->fill(0.0);
    }

    void Adam::optimize(const Tensor<float>& grads)const{
        // std::cout << "Optimize" << std::endl;
        // std::cout << "Weights: " << this->weights->shape() << " Grads: " << grads.shape() << " m: " << this->m->shape() << " v: " << this->v->shape() << std::endl;
        Tensor<float> m_hat(this->m->shape());
        Tensor<float> v_hat(this->v->shape());
        size_t count = this->weights->shape()[0];
        for(size_t i=0; i<count; i++){
            (*this->m)[i] = (*this->beta1)*(*this->m)[i] + ((float)(1.0)-(*this->beta1))*grads[i];
            (*this->v)[i] = (*this->beta2)*(*this->v)[i] + ((float)(1.0)-(*this->beta2))*grads[i]*grads[i];
            m_hat[i] = (*this->m)[i] / ((float)(1.0)-(*this->beta1));
            v_hat[i] = (*this->v)[i] / ((float)(1.0)-(*this->beta2));
            (*this->weights)[i] -= this->lr*m_hat[i]/(sqrt(v_hat[i])+1e-8);
        }
        if(this->bias != nullptr){
            (*this->m)[count] = (*this->beta1)*(*this->m)[count] + ((float)(1.0)-(*this->beta1))*grads[count];
            (*this->v)[count] = (*this->beta2)*(*this->v)[count] + ((float)(1.0)-(*this->beta2))*grads[count]*grads[count];
            m_hat[count] = (*this->m)[count] / ((float)(1.0)-(*this->beta1));
            v_hat[count] = (*this->v)[count] / ((float)(1.0)-(*this->beta2));
            (*this->bias) -= this->lr*m_hat[count]/(sqrt(v_hat[count])+1e-8);
        }
    }

    Adam::~Adam(){
        if(this->beta1 != nullptr)
            delete this->beta1;
        if(this->beta2 != nullptr)
            delete this->beta2;
        if(this->m != nullptr)
            delete this->m;
        if(this->v != nullptr)
            delete this->v;
    }
}