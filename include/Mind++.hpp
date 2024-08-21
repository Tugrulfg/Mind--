#ifndef MIND_PLUS_PLUS_HPP
#define MIND_PLUS_PLUS_HPP

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
        LinearRegression,
        BinaryLogisticRegression
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
        HuberLoss,
        RidgeLoss,
        LassoLoss,
        ElasticNetLoss,
        BCE,

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