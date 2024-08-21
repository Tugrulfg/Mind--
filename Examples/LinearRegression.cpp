#include "../include/Mind++.hpp"
#include <iostream>

using namespace cmind;

int main() {
    std::cout << "------------------------------- Linear Regression Testing -------------------------------" << std::endl;

    // Reading the csv file 
    CSVReader csv("./res/PolynomialRegression.csv", true);    

    // Creating the dataset from the csv data
    Dataset dataset(csv, 2, {}, 4, true);

    // Applying the necessary transforms to the dataset
    Transforms transforms(false, 3);    // Creating polynomial features of degree 3
    transforms.fit_transform(dataset);

    // Defining the loss and optimization functions
    HuberLoss huber(40000.0);
    Adam adam(0.05, 0.9, 0.999);

    // Defining the model
    LinearRegression lin_reg;
    lin_reg.compile(huber, adam);

    // Training the model
    lin_reg.train(dataset, 2000);

    // Loading the weights if there is a pretrained model
    // lin_reg.load("./Weights/PolynomialRegression.txt");

    // Testing the model
    std::vector<float> input = {58, 17};  // Example input
    std::vector<float> transformed = transforms(input);
    for(auto i: transformed)
        std::cout << i << " ";
    std::cout << "-> " << lin_reg.predict(transformed) << std::endl;

    // Saving the model
    lin_reg.save("./Weights/PolynomialRegression.txt");

    return 0;
}