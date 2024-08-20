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
            virtual void set_weights(Tensor<float>* weights, Tensor<float>* bias)=0;
        
        protected:
            // Type of the optimizer
            const Optimizers opt_type;

            // Learning rate
            Tensor<float> lr;

            // Weights
            Tensor<float>* weights = nullptr;

            // Bias
            Tensor<float>* bias = nullptr;
    };

    // Stochastic Gradient Descent(SGD) Optimizer
    class SGD: public Optimizer{
        public:
            SGD(const float lr);

            void set_weights(Tensor<float>* weights, Tensor<float>* bias = nullptr) override;

            void optimize(const Tensor<float>& grads)const override;
    };

    // Stochastic Gradient Descent(SGD) Optimizer with momentum
    class SGDMomentum: public Optimizer{
        public:
            SGDMomentum(const float lr, const float momentum);

            void optimize(const Tensor<float>& grads)const override;

            void set_weights(Tensor<float>* weights, Tensor<float>* bias = nullptr) override;

            ~SGDMomentum();

        private:
            Tensor<float>* momentum = nullptr;
            Tensor<float>* v = nullptr;
    };

    // AdaGrad Optimizer
    class AdaGrad: public Optimizer{
        public:
            AdaGrad(const float lr);

            void set_weights(Tensor<float>* weights, Tensor<float>* bias = nullptr) override;
        
            void optimize(const Tensor<float>& grads)const override;

            ~AdaGrad();
        private:
            Tensor<float>* v = nullptr;
    };

    // RMSProp Optimizer
    class RMSProp: public Optimizer{
        public:
            RMSProp(const float lr, const float beta);

            void set_weights(Tensor<float>* weights, Tensor<float>* bias = nullptr) override;

            void optimize(const Tensor<float>& grads)const override;

            ~RMSProp();
        private:
            Tensor<float>* beta = nullptr;
            Tensor<float>* v = nullptr;
    };

    // Adam Optimizer
    class Adam: public Optimizer{
        public:
            Adam(const float lr, const float beta1, const float beta2);

            void set_weights(Tensor<float>* weights, Tensor<float>* bias = nullptr) override;

            void optimize(const Tensor<float>& grads)const override;

            ~Adam();

        private:
            Tensor<float>* beta1 = nullptr;
            Tensor<float>* beta2 = nullptr;
            Tensor<float>* m = nullptr;
            Tensor<float>* v = nullptr;
    };



}

#endif