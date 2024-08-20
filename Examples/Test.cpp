#include "../include/Mind++.hpp"
#include <iostream>
#include <vector>
#include <tuple>

using namespace cmind;

int main() {
    // std::cout << "------------------------------- Matrix Multiplication Testing -------------------------------" << std::endl;
    
    // Tensor<float> in1({3,3});
    // Tensor<float> in2({3,3});

    // in1[0][0] = 1;
    // in1[0][1] = 2;
    // in1[0][2] = 3;
    // in1[1][0] = 1;
    // in1[1][1] = 2;
    // in1[1][2] = 1;
    // in1[2][0] = 3;
    // in1[2][1] = 1;
    // in1[2][2] = 2;

    // in2[0][0] = 1;
    // in2[0][1] = 2;
    // in2[0][2] = 3;
    // in2[1][0] = 1;
    // in2[1][1] = 2;
    // in2[1][2] = 1;
    // in2[2][0] = 3;
    // in2[2][1] = 1;
    // in2[2][2] = 2;

    // Tensor<float> out({3,3});
    // out.fill(1.0);
    // auto start = std::chrono::high_resolution_clock::now();

    // out += mat_mul(in1, in2);
    // auto end = std::chrono::high_resolution_clock::now();
    // auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
    // std::cout << time << std::endl;

    // std::cout << out << std::endl;
    // abort();

    // std::vector<float> input = {2.0, 3.0};  // Example input
    // int degree = 2;
    
    // std::vector<std::vector<float>> polyFeatures = createPolynomialFeatures(input, degree);
    
    // // Display the polynomial features
    // for (const auto& feature : polyFeatures) {
    //     std::cout << feature[0] << std::endl;
    // }

    // abort();

    // Shape shape({10,10});
    // Tensor<float> tensor1({10});
    // Tensor<float> tensor2({10, 10});

    // std::cout << "------------------------------- Shape Testing -------------------------------" << std::endl;
    // std::cout << tensor[0].shape << std::endl;
    // std::cout << tensor[0] << std::endl;


    // std::cout << "------------------------------- Tensor Testing -------------------------------" << std::endl;
    // std::cout << "Tensor1: \n" << tensor2 << std::endl;

    // std::cout << "Assign1" << std::endl;
    // for(int i=0; i<10; i++)
    //     tensor1[i] = i;

    // tensor2[0] += tensor1;
    // Tensor<float> tensor3(tensor2.copy());

    // std::cout << "Assign2" << std::endl;
    // std::cout << "Tensor2: \n" << tensor3 << std::endl;


    // std::cout << "------------------------------- CSVReader Testing -------------------------------" << std::endl

    // CSVReader csv("/home/tugrul/Desktop/Mind++/res/Salary_Data.csv", true);
    // std::cout << csv.shape() << std::endl;
    // std::cout << *(Tensor<std::string>*)std::get<0>(csv[5]) << std::endl;

    // std::cout << "------------------------------- Dataset Testing -------------------------------" << std::endl;

    // Dataset dataset(csv, 1, {}, 1, false);
    // std::cout << dataset << std::endl;
    // std::cout << csv << std::endl;

    // std::tuple <const void*, dtype> t = csv[0];
    // if(std::get<1>(t) == dtype::STR)
    //     std::cout << *(Tensor<char*>*)std::get<0>(t) << std::endl;
    // else if(std::get<1>(t) == dtype::INT)
    //     std::cout << *(Tensor<int>*)std::get<0>(t) << std::endl;
    // else if(std::get<1>(t) == dtype::FLOAT)
    //     std::cout << *(Tensor<float>*)std::get<0>(t) << std::endl;
    // else if(std::get<1>(t) == dtype::BOOL)
    //     std::cout << *(Tensor<bool>*)std::get<0>(t) << std::endl;
    // else 
    //     std::cout << "Error" << std::endl;

    // std::tuple<const void*, dtype> t = csv.at(149, 1);
    // if(std::get<1>(t) == dtype::STR)
    //     std::cout << *(char**)std::get<0>(t) << std::endl;
    // else if(std::get<1>(t) == dtype::INT)
    //     std::cout << *(int*)std::get<0>(t) << std::endl;
    // else if(std::get<1>(t) == dtype::FLOAT)
    //     std::cout << *(float*)std::get<0>(t) << std::endl;
    // else if(std::get<1>(t) == dtype::BOOL)
    //     std::cout << *(bool*)std::get<0>(t) << std::endl;
    // else 
    //     std::cout << "Error" << std::endl;

    // float val = 6.0;
    // csv.set(149, 1, &val);
    
    // if(std::get<1>(t) == dtype::STR)
    //     std::cout << *(char**)std::get<0>(t) << std::endl;
    // else if(std::get<1>(t) == dtype::INT)
    //     std::cout << *(int*)std::get<0>(t) << std::endl;
    // else if(std::get<1>(t) == dtype::FLOAT)
    //     std::cout << *(float*)std::get<0>(t) << std::endl;
    // else if(std::get<1>(t) == dtype::BOOL)
    //     std::cout << *(bool*)std::get<0>(t) << std::endl;
    // else 
    //     std::cout << "Error" << std::endl;


    // std::cout << "------------------------------- Loss Testing -------------------------------" << std::endl;

    // Tensor<float> tensor1({1,2});
    // Tensor<float> tensor2({2});
    // Tensor<float> tensor3({1});
    // Tensor<float> tensor4({2});

    // tensor1[0][0] = 1.0;
    // tensor1[0][1] = 2.0;
    // tensor2[0] = 5.0;
    // tensor2[1] = 6.0;

    // std::cout << "Model" << std::endl;
    // LinearRegression lin_reg(2);
    // std::cout << "Model" << std::endl;
    
    // tensor3 = lin_reg(tensor1);

    // std::cout << "Tensor3: " << tensor3 << std::endl;
    // std::cout << "Weights: " << *lin_reg.weights() << std::endl;

    // MSE mse(Algorithms::LinearRegression);

    // SGD sgd(0.001);

    // tensor3 = mse.compute(tensor1, tensor2);
    // tensor4 = mse.gradient(tensor1, tensor2);

    // std::cout << "Loss: " << tensor3 << std::endl;
    // std::cout << "Gradients: " << tensor4 << std::endl;
    

    std::cout << "------------------------------- Training Testing -------------------------------" << std::endl;

    // CSVReader csv("/home/tugrul/Desktop/Mind++/Examples/res/Linear_test.csv", true);
    // Dataset dataset(csv, 1, {}, 4, false);
    


    // std::cout << dataset << std::endl;
    // abort();

    // CSVReader csv("/home/tugrul/Desktop/Mind++/Examples/res/Income.csv", true);
    // Dataset dataset(csv, 2, {}, 4, false);

    // CSVReader csv("/home/tugrul/Desktop/Mind++/Examples/res/Iris.csv", true);
    // Dataset dataset(csv, 4, {0}, 4, false);

    CSVReader csv("/home/tugrul/Desktop/Mind++/Examples/res/Student.csv", true);
    Dataset dataset(csv, 5, {}, 4, true);

    Transforms transforms(true, 1);
    transforms.fit_transform(dataset);

    // std::cout << dataset << std::endl;
    // abort();

    MAE mae;
    MSE mse;
    HuberLoss huber(50000.0);
    RidgeLoss ridge(0.1);
    LassoLoss lasso(0.1);
    ElasticNetLoss elastic_net(0.1, 0.5);

    SGDMomentum sgdm(0.001, 0.9);
    SGD sgd(0.001);
    AdaGrad adagrad(0.1);
    RMSProp rmsprop(0.001, 0.9);
    Adam adam(0.01, 0.9, 0.999);
    
    LinearRegression lin_reg(dataset.shape()[1]);
    // lin_reg.load("/home/tugrul/Desktop/Mind++/Examples/Weights/Student.txt");
    
    lin_reg.compile(elastic_net, adam);
    lin_reg.train(dataset, 10);

    // lin_reg.save("/home/tugrul/Desktop/Mind++/Examples/Weights/Student.txt");

    std::vector<float> input = {9, 97, 1, 7, 0};  // Example input
    std::vector<float> transformed = transforms(input);
    for(auto i: transformed)
        std::cout << i << " ";
    std::cout << "-> " << lin_reg.predict(transformed) << std::endl;
    // std::cout << "Pred 12.0: " << lin_reg.predict({12.}) << std::endl;
    // std::cout << "Pred 13.0: " << lin_reg.predict({13.}) << std::endl;
    // std::cout << "Pred 14.0: " << lin_reg.predict({14.}) << std::endl;
    // std::cout << "Pred 15.0: " << lin_reg.predict({15.}) << std::endl;
    // std::cout << "Pred 16.0: " << lin_reg.predict({16.}) << std::endl;
    // std::cout << "Pred 17.0: " << lin_reg.predict({17.}) << std::endl;
    // std::cout << "Pred 18.0: " << lin_reg.predict({18.}) << std::endl;


    // std::cout << "Weights: " << lin_reg.weights()->flatten() << std::endl;
    std::cout << "Destructor Calls" << std::endl;

    return 0;
}