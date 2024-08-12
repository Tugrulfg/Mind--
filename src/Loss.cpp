#include "../include/Loss.hpp"

namespace cmind{

    // Constructor
    Loss::Loss(const Algorithms alg_type, const Losses loss_type): alg_type(alg_type), loss_type(loss_type){

    }

    // Set the weight count
    void Loss::set_weight_count(const size_t count){
        this->weight_count = count;
    }

    // Set the inputs
    void Loss::set_inputs(Tensor<float>* inputs){
        this->inputs = inputs;
    }

    // Destructor
    Loss::~Loss(){
        if(this->inputs != nullptr)
            delete this->inputs;
    }


    // MSE constructor
    MSE::MSE(const Algorithms alg_type): Loss(alg_type, Losses::MSE){

    }

    // Calculates the loss value according to the given pred and target
    const Tensor<float> MSE::compute(const Tensor<float>& pred, const Tensor<float>& target)const{
        if(pred.shape() != target.shape()){
            std::cout << "MSE Loss: Shapes do not match" << std::endl;
            abort();
        }
        else if(pred.shape().size() != 1){
            std::cout << "MSE Loss: Invalid shape: " << pred.shape() << std::endl;
            abort();
        }
        Tensor<float> loss({1});
        for(size_t i=0; i<pred.shape()[0]; i++)
            loss += (pred[i]-target[i])*(pred[i]-target[i]);
        return loss/(2*pred.shape()[0]);
    }

    // Calculates the gradients of the variables
    const Tensor<float> MSE::gradient(const Tensor<float>& pred, const Tensor<float>& target)const{
        // std::cout << "Pred: " << pred.shape() << "Target: " << target.shape() << "Weights: " << this->weight_count << "Inputs: " << this->inputs->shape() << std::endl;
        if(pred.shape() != target.shape()){
            std::cout << "MSE Loss: Shapes do not match" << std::endl;
            abort();
        }
        else if(pred.shape().size() != 1){
            std::cout << "MSE Loss: Invalid shape: " << pred.shape() << std::endl;
            abort();
        }        
        size_t batch_size = pred.shape()[0];
        Tensor<float> grads({this->weight_count});
        grads.fill(0.0);
        size_t count = this->weight_count;
        if(this->alg_type == Algorithms::LinearRegression)      // Decrement for the bias
            count--;

        // Calculate gradients
        for(size_t i=0; i<batch_size; i++){
            for(size_t j=0; j<count; j++)
                grads[j] += (pred[i]-target[i])*(*this->inputs)[i][j]/batch_size;
            
            if(this->alg_type == Algorithms::LinearRegression)
                grads[count] += (pred[i]-target[i])/batch_size;
        }
        
        return grads;
    }


    // MAE constructor
    MAE::MAE(const Algorithms alg_type): Loss(alg_type, Losses::MAE){
        
    }

    // Calculates the loss value according to the given pred and target
    const Tensor<float> MAE::compute(const Tensor<float>& pred, const Tensor<float>& target)const{
        if(pred.shape() != target.shape()){
            std::cout << "MAE Loss: Shapes do not match" << std::endl;
            abort();
        }
        else if(pred.shape().size() != 1){
            std::cout << "MAE Loss: Invalid shape: " << pred.shape() << std::endl;
            abort();
        }
        Tensor<float> loss({1});
        float val = abs_dif(pred, target).mean();
        loss[0] = val;
        return loss;
    }

    // Calculates the gradients of the variables
    const Tensor<float> MAE::gradient(const Tensor<float>& pred, const Tensor<float>& target)const{
        if(pred.shape() != target.shape()){
            std::cout << "MAE Loss: Shapes do not match" << std::endl;
            abort();
        }
        else if(pred.shape().size() != 1){
            std::cout << "MAE Loss: Invalid shape: " << pred.shape() << std::endl;
            abort();
        }
        size_t batch_size = pred.shape()[0];
        Tensor<float> grads({this->weight_count});
        grads.fill(0.0);
        size_t count = this->weight_count;
        if(this->alg_type == Algorithms::LinearRegression)      // Decrement for the bias
            count--;

        // Calculate gradients
        for(size_t i=0; i<batch_size; i++){
            if((pred[i] - target[i]).all_negative()){
                for(size_t j=0; j<count; j++)
                    grads[j] -= (*this->inputs)[i][j]/batch_size;
                
                if(this->alg_type == Algorithms::LinearRegression)
                    grads[count] -= 1.0/batch_size;
            }
            else{
                for(size_t j=0; j<count; j++)
                    grads[j] += (*this->inputs)[i][j]/batch_size;
                
                if(this->alg_type == Algorithms::LinearRegression)
                    grads[count] += 1.0/batch_size;
            }
            
        }
        
        return grads;
    }
}