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

            // Returns the loss function type
            const Losses get_loss_type()const;

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

    // Ridge Loss (L2) implementation
    class RidgeLoss: public Loss{
        public:
            RidgeLoss(const float alpha);
            const Tensor<float> compute(const Tensor<float>& pred, const Tensor<float>& target)const override;
            const Tensor<float> gradient(const Tensor<float>& pred, const Tensor<float>& target)const override;

            // Set the weights
            void set_weights(Tensor<float>* weights, Tensor<float>* bias);

            ~RidgeLoss();
        private:
            Tensor<float>* alpha = nullptr; 
            Tensor<float>* weights = nullptr;
            Tensor<float>* bias = nullptr;
    };

    // Lasso Loss (L1) implementation
    class LassoLoss: public Loss{
        public:
            LassoLoss(const float alpha);
            const Tensor<float> compute(const Tensor<float>& pred, const Tensor<float>& target)const override;
            const Tensor<float> gradient(const Tensor<float>& pred, const Tensor<float>& target)const override;

            // Set the weights
            void set_weights(Tensor<float>* weights, Tensor<float>* bias);

            ~LassoLoss();

        private:
            Tensor<float>* alpha = nullptr; 
            Tensor<float>* weights = nullptr;
            Tensor<float>* bias = nullptr;
    };

    // ElasticNet Loss implementation
    class ElasticNetLoss: public Loss{
        public:
            ElasticNetLoss(const float alpha, const float l1_ratio);
            const Tensor<float> compute(const Tensor<float>& pred, const Tensor<float>& target)const override;
            const Tensor<float> gradient(const Tensor<float>& pred, const Tensor<float>& target)const override;

            // Set the weights
            void set_weights(Tensor<float>* weights, Tensor<float>* bias);

            ~ElasticNetLoss();

        private:
            Tensor<float>* alpha = nullptr; 
            Tensor<float>* l1_ratio = nullptr;
            Tensor<float>* weights = nullptr;
            Tensor<float>* bias = nullptr;
    };

    // Binary Cross Entropy Loss implementation
    class BCE: public Loss{
        public:
            BCE();
            const Tensor<float> compute(const Tensor<float>& pred, const Tensor<float>& target)const override;
            const Tensor<float> gradient(const Tensor<float>& pred, const Tensor<float>& target)const override;
    };
}

#endif