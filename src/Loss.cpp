#include "../include/Loss.hpp"

namespace cmind{

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
        return loss/pred.shape()[0];
    }

    // Calculates the gradients of the variables
    const Tensor<float> MSE::gradient(const Tensor<float>& pred, const Tensor<float>& target)const{
        if(pred.shape() != target.shape()){
            std::cout << "MSE Loss: Shapes do not match" << std::endl;
            abort();
        }
        else if(pred.shape().size() != 1){
            std::cout << "MSE Loss: Invalid shape: " << pred.shape() << std::endl;
            abort();
        }
        Tensor<float> grads(pred.shape());
        for(size_t i=0; i<pred.shape()[0]; i++)
            grads[i] = (pred[i]-target[i])/pred.shape()[0];
        return grads;
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
        float val = abs(pred, target).mean();
        loss = val;
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
        Tensor<float> grads(pred.shape());
        const float* hat = pred.data();
        const float* tar = target.data();
        for(size_t i=0; i<pred.shape()[0]; i++){
            if(hat[i] > tar[i])
                grads[i] = 1. / pred.shape()[0];
            else
                grads[i] = -1. / pred.shape()[0];
        }
        return grads;
    }
}