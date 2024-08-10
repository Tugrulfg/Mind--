#ifndef LOSS_HPP
#define LOSS_HPP

#include "Mind++.hpp"

namespace cmind{
    // Base Loss Class
    class Loss{
        public:

            // Calculates the loss value according to the given pred and target
            virtual const Tensor<float> compute(const Tensor<float>& pred, const Tensor<float>& target)const = 0;
        
            // Calculates the gradients of the variables
            virtual const Tensor<float> gradient(const Tensor<float>& pred, const Tensor<float>& target)const = 0;
    };

    // Mean Squared Error(MSE) Loss implementation
    class MSE: public Loss{
        public:
            const Tensor<float> compute(const Tensor<float>& pred, const Tensor<float>& target)const override;
            const Tensor<float> gradient(const Tensor<float>& pred, const Tensor<float>& target)const override;
    };

    // Mean Absolute Error(MAE) Loss implementation
    class MAE: public Loss{
        public:
            const Tensor<float> compute(const Tensor<float>& pred, const Tensor<float>& target)const override;
            const Tensor<float> gradient(const Tensor<float>& pred, const Tensor<float>& target)const override;
    };
}

#endif