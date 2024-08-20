#include "../include/Loss.hpp"

namespace cmind{

    // Constructor
    Loss::Loss(const Losses loss_type): loss_type(loss_type){

    }

    // Set the weight count
    void Loss::set_weight_count(const size_t count){
        this->weight_count = count;
    }

    // Set the inputs
    void Loss::set_inputs(Tensor<float>* inputs){
        this->inputs = inputs;
    }

    // Set the algorithm type
    void Loss::set_alg_type(const Algorithms alg_type){
        this->alg_type = alg_type;
    }

    // Returns the loss function type
    const Losses Loss::get_loss_type()const{
        return this->loss_type;
    }

    // Destructor
    Loss::~Loss(){
        
    }


    // MSE constructor
    MSE::MSE(): Loss(Losses::MSE){

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
        loss.fill(0.0);
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
        size_t count = 0;
        if(this->alg_type == Algorithms::LinearRegression)
            count = this->weight_count+1;
        else
            count = this->weight_count;
        Tensor<float> grads({count});
        grads.fill(0.0);

        if(this->alg_type == Algorithms::LinearRegression)
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
    MAE::MAE(): Loss(Losses::MAE){
        
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
        loss.fill(0.0);
        loss[0] = abs(pred - target).mean();
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
        size_t count = 0;
        if(this->alg_type == Algorithms::LinearRegression)
            count = this->weight_count+1;
        else
            count = this->weight_count;
        Tensor<float> grads({count});
        grads.fill(0.0);
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


    // HuberLoss constructor
    HuberLoss::HuberLoss(const float delta): Loss(Losses::HuberLoss){
        this->delta = new Tensor<float>({1});
        this->delta->fill(delta);
    }

    // Calculates the loss value according to the given pred and target
    const Tensor<float> HuberLoss::compute(const Tensor<float>& pred, const Tensor<float>& target)const{
        if(pred.shape() != target.shape()){
            std::cout << "Huber Loss: Shapes do not match" << std::endl;
            abort();
        }
        else if(pred.shape().size() != 1){
            std::cout << "Huber Loss: Invalid shape: " << pred.shape() << std::endl;
            abort();
        }
        Tensor<float> loss({1});
        loss.fill(0.0);
        Tensor<float> diff({1});

        for(size_t i=0; i<pred.shape()[0]; i++){
            diff = abs(pred[i] - target[i]);
            if(diff <= this->delta[0])
                loss[0] += diff*diff/2.0;
            else
                loss[0] += this->delta[0]*(diff-this->delta[0]/2.0);
        }
        return loss/pred.shape()[0];
    }

    // Calculates the gradients of the variables
    const Tensor<float> HuberLoss::gradient(const Tensor<float>& pred, const Tensor<float>& target)const{
        if(pred.shape() != target.shape()){
            std::cout << "Huber Loss: Shapes do not match" << std::endl;
            abort();
        }
        else if(pred.shape().size() != 1){
            std::cout << "Huber Loss: Invalid shape: " << pred.shape() << std::endl;
            abort();
        }
        size_t batch_size = pred.shape()[0];
        size_t count = 0;
        if(this->alg_type == Algorithms::LinearRegression)
            count = this->weight_count+1;
        else
            count = this->weight_count;
        Tensor<float> grads({count});
        grads.fill(0.0);
        if(this->alg_type == Algorithms::LinearRegression)      // Decrement for the bias
            count--;

        Tensor<float> diff({1});

        // Calculate gradients  
        for(size_t i=0; i<batch_size; i++){
            diff = abs(pred[i] - target[i]);
            if(diff <= this->delta[0]){
                for(size_t j=0; j<count; j++)
                    grads[j] += (pred[i]-target[i])*(*this->inputs)[i][j]/batch_size;
                
                if(this->alg_type == Algorithms::LinearRegression)
                    grads[count] += (pred[i]-target[i])/batch_size;
            }
            else{
                if((pred[i] - target[i]).all_negative()){
                    for(size_t j=0; j<count; j++)
                        grads[j] -= (*this->inputs)[i][j]/batch_size;
                    
                    if(this->alg_type == Algorithms::LinearRegression)
                        grads[count] -= 1.0/batch_size;
                }
                else{
                    for(size_t j=0; j<count; j++)
                        grads[j] += (*this->inputs)[i][j]*this->delta[0]/batch_size;
                    
                    if(this->alg_type == Algorithms::LinearRegression)
                        grads[count] += this->delta[0]*1.0/batch_size;
                }
            }
            
        }
        return grads;
    }

    // Destructor
    HuberLoss::~HuberLoss(){
        if(this->delta != nullptr)
            delete this->delta;
    }


    RidgeLoss::RidgeLoss(const float alpha): Loss(Losses::RidgeLoss){
        this->alpha = new Tensor<float>({1});
        this->alpha->fill(alpha);
    }

    // Calculates the loss value according to the given pred and target
    const Tensor<float> RidgeLoss::compute(const Tensor<float>& pred, const Tensor<float>& target)const{
        if(pred.shape() != target.shape()){
            std::cout << "Ridge Loss: Shapes do not match" << std::endl;
            abort();
        }
        else if(pred.shape().size() != 1){
            std::cout << "Ridge Loss: Invalid shape: " << pred.shape() << std::endl;
            abort();
        }
        Tensor<float> loss({1});
        loss.fill(0.0);
        for(size_t i=0; i<pred.shape()[0]; i++)
            loss += (pred[i]-target[i])*(pred[i]-target[i]);
        loss = loss/(2*pred.shape()[0]);

        Tensor<float> penalty({1});
        penalty.fill(0.0);
        for(size_t i=0; i<this->weight_count; i++)
            penalty += power((*this->weights)[i], 2);
        penalty += power((*this->bias)[0], 2);
        penalty *= this->alpha[0]/((this->weight_count+1)*2.0);
        return loss+penalty;
    }

    // Calculates the gradients of the variables
    const Tensor<float> RidgeLoss::gradient(const Tensor<float>& pred, const Tensor<float>& target)const{
        // std::cout << "Pred: " << pred.shape() << "Target: " << target.shape() << "Weights: " << this->weight_count << "Inputs: " << this->inputs->shape() << std::endl;
        if(pred.shape() != target.shape()){
            std::cout << "Ridge Loss: Shapes do not match" << std::endl;
            abort();
        }
        else if(pred.shape().size() != 1){
            std::cout << "Ridge Loss: Invalid shape: " << pred.shape() << std::endl;
            abort();
        }        
        size_t batch_size = pred.shape()[0];
        size_t count = 0;
        if(this->alg_type == Algorithms::LinearRegression)
            count = this->weight_count+1;
        else
            count = this->weight_count;
        Tensor<float> grads({count});
        grads.fill(0.0);

        // Calculate gradients
        for(size_t i=0; i<batch_size; i++){
            for(size_t j=0; j<this->weight_count; j++)
                grads[j] += ((pred[i]-target[i])*(*this->inputs)[i][j]/batch_size + (*this->weights)[j]*this->alpha[0]/(this->weight_count+1));
            
            if(this->bias != nullptr)
                grads[this->weight_count] += ((pred[i]-target[i])/batch_size + (*this->bias)[0]*this->alpha[0]/(this->weight_count+1));
        }
        
        return grads;
    }

    // Set the weights
    void RidgeLoss::set_weights(Tensor<float>* weights, Tensor<float>* bias){
        this->weights = weights;
        this->bias = bias;
    }

    // Destructor
    RidgeLoss::~RidgeLoss(){
        if(this->alpha != nullptr)
            delete this->alpha;
    }


    LassoLoss::LassoLoss(const float alpha): Loss(Losses::LassoLoss){
        this->alpha = new Tensor<float>({1});
        this->alpha->fill(alpha);
    }

    // Calculates the loss value according to the given pred and target
    const Tensor<float> LassoLoss::compute(const Tensor<float>& pred, const Tensor<float>& target)const{
        if(pred.shape() != target.shape()){
            std::cout << "Ridge Loss: Shapes do not match" << std::endl;
            abort();
        }
        else if(pred.shape().size() != 1){
            std::cout << "Ridge Loss: Invalid shape: " << pred.shape() << std::endl;
            abort();
        }
        Tensor<float> loss({1});
        loss.fill(0.0);
        for(size_t i=0; i<pred.shape()[0]; i++)
            loss += (pred[i]-target[i])*(pred[i]-target[i]);
        loss = loss/(2*pred.shape()[0]);

        Tensor<float> penalty({1});
        penalty.fill(0.0);
        for(size_t i=0; i<this->weight_count; i++)
            penalty += abs((*this->weights)[i]);
        penalty += abs((*this->bias)[0]);
        penalty *= this->alpha[0]/((this->weight_count+1)*2.0);
        return loss+penalty;
    }

    // Calculates the gradients of the variables
    const Tensor<float> LassoLoss::gradient(const Tensor<float>& pred, const Tensor<float>& target)const{
        // std::cout << "Pred: " << pred.shape() << "Target: " << target.shape() << "Weights: " << this->weight_count << "Inputs: " << this->inputs->shape() << std::endl;
        if(pred.shape() != target.shape()){
            std::cout << "Ridge Loss: Shapes do not match" << std::endl;
            abort();
        }
        else if(pred.shape().size() != 1){
            std::cout << "Ridge Loss: Invalid shape: " << pred.shape() << std::endl;
            abort();
        }        
        size_t batch_size = pred.shape()[0];
        Tensor<float> grads({this->weight_count+1});
        grads.fill(0.0);

        // Calculate gradients
        for(size_t i=0; i<batch_size; i++){
            for(size_t j=0; j<this->weight_count; j++){
                if((*this->weights)[j] < 0)
                    grads[j] += ((pred[i]-target[i])*(*this->inputs)[i][j]/(batch_size*2.0) - (*this->alpha)/((this->weight_count+1)*2.0));
                else
                    grads[j] += ((pred[i]-target[i])*(*this->inputs)[i][j]/(batch_size*2.0) + (*this->alpha)/((this->weight_count+1)*2.0));
            }
            
            if(this->bias != nullptr){
                if((*this->bias)[0] < 0)
                    grads[this->weight_count] += (pred[i]-target[i])/(batch_size*2.0) - (*this->alpha)/((this->weight_count+1)*2.0);
                else
                    grads[this->weight_count] += (pred[i]-target[i])/(batch_size*2.0) + (*this->alpha)/((this->weight_count+1)*2.0);
            }

        }
        
        return grads;
    }

    // Set the weights
    void LassoLoss::set_weights(Tensor<float>* weights, Tensor<float>* bias){
        this->weights = weights;
        this->bias = bias;
    }

    // Destructor
    LassoLoss::~LassoLoss(){
        if(this->alpha != nullptr)
            delete this->alpha;
    }


    ElasticNetLoss::ElasticNetLoss(const float alpha, const float l1_ratio): Loss(Losses::ElasticNetLoss){
        this->alpha = new Tensor<float>({1});
        this->alpha->fill(alpha);

        this->l1_ratio = new Tensor<float>({1});
        this->l1_ratio->fill(l1_ratio);
    }

    // Calculates the loss value according to the given pred and target
    const Tensor<float> ElasticNetLoss::compute(const Tensor<float>& pred, const Tensor<float>& target)const{
        if(pred.shape() != target.shape()){
            std::cout << "Ridge Loss: Shapes do not match" << std::endl;
            abort();
        }
        else if(pred.shape().size() != 1){
            std::cout << "Ridge Loss: Invalid shape: " << pred.shape() << std::endl;
            abort();
        }
        Tensor<float> loss({1});
        loss.fill(0.0);
        for(size_t i=0; i<pred.shape()[0]; i++)
            loss += (pred[i]-target[i])*(pred[i]-target[i]);
        loss = loss/(2*pred.shape()[0]);

        Tensor<float> penalty({1});
        penalty.fill(0.0);
        for(size_t i=0; i<this->weight_count; i++)
            penalty += (*this->l1_ratio * abs((*this->weights)[i]) + (-(*this->l1_ratio)+1) * power((*this->weights)[i], 2)/2.0);
        penalty += (*this->l1_ratio * abs((*this->bias)[0]) + (-(*this->l1_ratio)+1) * power((*this->bias)[0], 2)/2.0);
        penalty *= this->alpha[0]/this->weight_count;
        return loss+penalty;
    }

    // Calculates the gradients of the variables
    const Tensor<float> ElasticNetLoss::gradient(const Tensor<float>& pred, const Tensor<float>& target)const{
        // std::cout << "Pred: " << pred.shape() << "Target: " << target.shape() << "Weights: " << this->weight_count << "Inputs: " << this->inputs->shape() << std::endl;
        if(pred.shape() != target.shape()){
            std::cout << "Ridge Loss: Shapes do not match" << std::endl;
            abort();
        }
        else if(pred.shape().size() != 1){
            std::cout << "Ridge Loss: Invalid shape: " << pred.shape() << std::endl;
            abort();
        }        
        size_t batch_size = pred.shape()[0];
        Tensor<float> grads({this->weight_count+1});
        grads.fill(0.0);        

        // Calculate gradients
        for(size_t i=0; i<batch_size; i++){
            for(size_t j=0; j<this->weight_count; j++){
                if((*this->weights)[j] < 0)
                    grads[j] += ((pred[i]-target[i])*(*this->inputs)[i][j]/batch_size + (*this->alpha/((this->weight_count+1)*2.0))*(*this->alpha*(-1) + (-*this->alpha+1)*(*this->weights)[j]));
                else
                    grads[j] += ((pred[i]-target[i])*(*this->inputs)[i][j]/batch_size + (*this->alpha/((this->weight_count+1)*2.0))*(*this->alpha*1 + (-*this->alpha+1)*(*this->weights)[j]));
            }
            
            if(this->bias != nullptr){
                if((*this->bias)[0] < 0)
                    grads[this->weight_count] += ((pred[i]-target[i])/batch_size + (*this->alpha/((this->weight_count+1)*2.0))*(*this->alpha*(-1) + (-*this->alpha+1)*(*this->bias)[0]));
                else
                    grads[this->weight_count] += ((pred[i]-target[i])/batch_size + (*this->alpha/((this->weight_count+1)*2.0))*(*this->alpha*1 + (-*this->alpha+1)*(*this->bias)[0]));
            }
        }
        
        return grads;
    }

    // Set the weights
    void ElasticNetLoss::set_weights(Tensor<float>* weights, Tensor<float>* bias){
        this->weights = weights;
        this->bias = bias;
    }

    // Destructor
    ElasticNetLoss::~ElasticNetLoss(){
        if(this->alpha != nullptr)
            delete this->alpha;
        if(this->l1_ratio != nullptr)
            delete this->l1_ratio;
    }

}