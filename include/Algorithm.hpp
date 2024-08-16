#ifndef ALGORITHM_HPP
#define ALGORITHM_HPP

#include "Mind++.hpp"
#include <fstream>
#include <string>
#include <chrono>

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

            // Predicts for a single input
            virtual const Tensor<float> predict(const std::vector<float> input)const = 0;

            // Returns the weights
            const Tensor<float>* weights()const;

            // Saves the weights
            virtual void save(const std::string& filepath)const =0;

            // Loads the weights
            virtual void load(const std::string& filepath)=0;

        protected:

            // Type of the machine learning algorithm
            const Algorithms alg_type;

            // Trainable weights of the algorithm
            Tensor<float>* weights_;
    };

    // Linear regression algorithm implementation
    class LinearRegression: public Algorithm{
        public:
            LinearRegression(const size_t input_count);

            // Load model from file
            LinearRegression(const std::string& filepath);

            const Tensor<float> operator()(const Tensor<float>& input)const override;

            void train(Dataset& ds, const size_t epochs, Loss& loss_func, Optimizer& opt) override;

            const Tensor<float> predict(const std::vector<float> input)const override;

            void save(const std::string& filepath)const override;

            void load(const std::string& filepath)override;

            // Destructor
            ~LinearRegression();
    };

    // Polynomial regression algorithm implementation
    // class PolynomialRegression: public Algorithm{
    //     public:
    //         PolynomialRegression(const size_t input_count, const size_t degree);

    //         // Load model from file
    //         PolynomialRegression(const std::string& filepath);

    //         const Tensor<float> operator()(const Tensor<float>& input)const override;

    //         void train(Dataset& ds, const size_t epochs, Loss& loss_func, Optimizer& opt) override;

    //         const Tensor<float> predict(const std::vector<float> input)const override;

    //          void save(const std::string& filepath)const override;

    //         void load(const std::string& filepath)override;

    //         // Destructor
    //         ~PolynomialRegression();
    //     private:
    //         Tensor<float>* num_var;
    //         Tensor<float>* degree;
    // };
}







#endif