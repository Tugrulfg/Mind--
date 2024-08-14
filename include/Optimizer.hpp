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
            virtual void set_weights(Tensor<float>* weights)=0;
        
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

            void set_weights(Tensor<float>* weights) override;

            void optimize(const Tensor<float>& grads)const override;
    };

    // Stochastic Gradient Descent(SGD) Optimizer with momentum
    class SGDMomentum: public Optimizer{
        public:
            SGDMomentum(const float lr, const float momentum);

            void optimize(const Tensor<float>& grads)const override;

            void set_weights(Tensor<float>* weights) override;

            ~SGDMomentum();

        private:
            Tensor<float>* momentum = nullptr;
            Tensor<float>* velocity = nullptr;
    };

}

#endif