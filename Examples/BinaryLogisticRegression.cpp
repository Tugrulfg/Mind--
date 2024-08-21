#include "../include/Mind++.hpp"
#include <iostream>

using namespace cmind;

int main() {
    std::cout << "------------------------------- Binary Logistic Regression Testing -------------------------------" << std::endl;

    // Reading the csv file 
    CSVReader csv("./res/BinaryClassification.csv", true);    

    // Creating the dataset from the csv data
    Dataset dataset(csv, 4, {0}, 4, false);

    // Applying the necessary transforms to the dataset
    Transforms transforms(false, 1);   
    transforms.fit_transform(dataset);

    // Defining the loss and optimization functions
    BCE bce;
    Adam adam(0.0002, 0.9, 0.999);

    // Defining the model
    BinaryLogisticRegression log_reg;
    log_reg.compile(bce, adam);

    // Training the model
    log_reg.train(dataset, 2000);

    // Loading the weights if there is a pretrained model
    // log_reg.load("./Weights/BinaryLogisticRegression.txt");

    // Testing the model
    std::vector<float> input = {0, 19, 19000};  // Example input
    std::vector<float> transformed = transforms(input);
    for(auto i: transformed)
        std::cout << i << " ";
    std::cout << "-> " << log_reg.predict(transformed) << std::endl;

    // Saving the model
    log_reg.save("./Weights/BinaryLogisticRegression.txt");

    return 0;

}