#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "Mind++.hpp"

namespace cmind{

    // Base Optimizer class
    class Optimizer{
        public:
            // Constructor
            Optimizer(const Optimizers opt_type, const float lr = 0.01);

            // Optimizes the weights
            virtual void optimize(const Tensor<float>& grads)const = 0;

            // Set the weights
            void set_weights(Tensor<float>* weights);
        
        protected:
            // Type of the optimizer
            const Optimizers opt_type;

            // Learning rate
            Tensor<float> lr;

            // Weights
            Tensor<float>* weights;
    };

    // Stochastic Gradient Descent(SGD) Optimizer
    class SGD: public Optimizer{
        public:
            SGD(const float lr);

            void optimize(const Tensor<float>& grads)const override;
    };

}

#endif