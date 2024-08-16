#ifndef LOSS_HPP
#define LOSS_HPP

#include "Mind++.hpp"

namespace cmind{
    // Base Loss Class
    class Loss{
        public:
            // Constructor
            Loss(const Losses loss_type);

            // Calculates the loss value according to the given pred and target
            virtual const Tensor<float> compute(const Tensor<float>& pred, const Tensor<float>& target)const = 0;
        
            // Calculates the gradients of the variables
            virtual const Tensor<float> gradient(const Tensor<float>& pred, const Tensor<float>& target)const = 0;

            // Set the weight count
            void set_weight_count(const size_t count);

            // Set the inputs
            void set_inputs(Tensor<float>* inputs);

            // Set the algorithm type
            void set_alg_type(const Algorithms alg_type);

            // Destructor
            ~Loss();
        
        protected:
            // Type of the machine learning algorithm
            Algorithms alg_type; 

            // Trainable weights of the algorithm
            size_t weight_count;

            // Loss function type
            const Losses loss_type;

            // Inputs
            Tensor<float>* inputs = nullptr;
    };

    // Mean Squared Error(MSE) Loss implementation
    class MSE: public Loss{
        public:
            MSE();
            const Tensor<float> compute(const Tensor<float>& pred, const Tensor<float>& target)const override;
            const Tensor<float> gradient(const Tensor<float>& pred, const Tensor<float>& target)const override;
    };

    // Mean Absolute Error(MAE) Loss implementation
    class MAE: public Loss{
        public:
            MAE();
            const Tensor<float> compute(const Tensor<float>& pred, const Tensor<float>& target)const override;
            const Tensor<float> gradient(const Tensor<float>& pred, const Tensor<float>& target)const override;
    };

    // Huber Loss implementation
    class HuberLoss: public Loss{
        public:
            HuberLoss(const float delta);
            const Tensor<float> compute(const Tensor<float>& pred, const Tensor<float>& target)const override;
            const Tensor<float> gradient(const Tensor<float>& pred, const Tensor<float>& target)const override;

            ~HuberLoss();
        private:
            Tensor<float>* delta = nullptr; 
    };
}

#endif