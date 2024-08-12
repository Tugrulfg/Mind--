#ifndef ALGORITHM_HPP
#define ALGORITHM_HPP

#include "Mind++.hpp"

namespace cmind{
    class Loss;
    class Optimizer;
    class Dataset;

    class Algorithm{
        public:
            // Constructor
            Algorithm(const Algorithms alg_type);

            // Runs the algorithm and returns the predictions
            virtual const Tensor<float> operator()(const Tensor<float>& input)const = 0;

            // Trains the algorithm 
            virtual void train(Dataset& ds, const size_t epochs, Loss& loss_func, Optimizer& opt) = 0;

            // Returns the weights
            const Tensor<float>* weights()const;

        protected:
            // Type of the machine learning algorithm
            const Algorithms alg_type;

            // Trainable weights of the algorithm
            Tensor<float>* weights_;
    };

    class LinearRegression: public Algorithm{
        public:
            LinearRegression(const size_t input_count);

            const Tensor<float> operator()(const Tensor<float>& input)const override;

            void train(Dataset& ds, const size_t epochs, Loss& loss_func, Optimizer& opt) override;

            // Destructor
            ~LinearRegression();
    };
}







#endif