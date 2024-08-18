#ifndef C_PLUS_PLUS_MIND_HPP
#define C_PLUS_PLUS_MIND_HPP

namespace cmind{
    // Enum type for data types
    enum class dtype{
        BOOL,
        INT,
        FLOAT,
        STR
    }; 

    // Enum type for machine learning algorithms
    enum class Algorithms{
        LinearRegression
    }; 

    // Enum type for optimizers
    enum class Optimizers{
        SGD,
        SGDMomentum, 
        AdaGrad,
        RMSProp,
        Adam
    };

    // Enum type for losses
    enum class Losses{
        MSE,
        MAE,
        HuberLoss
    };
}

#include "Tensor.hpp"
#include "Shape.hpp"
#include "CSVReader.hpp"
#include "Dataset.hpp"
#include "Loss.hpp"
#include "Math.hpp"
#include "Optimizer.hpp"
#include "Algorithm.hpp"
#include "Utils.hpp"
#include "Transforms.hpp"

#endif