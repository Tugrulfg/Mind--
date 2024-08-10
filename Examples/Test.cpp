#include "../include/Mind++.hpp"
#include <iostream>
#include <vector>
#include <tuple>

using namespace cmind;

int main(){

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


    std::cout << "------------------------------- CSVReader Testing -------------------------------" << std::endl;

    CSVReader csv("/home/tugrul/Desktop/Mind++/res/Iris.csv", true);
    std::cout << csv.shape() << std::endl;
    // std::cout << *(Tensor<std::string>*)std::get<0>(csv[5]) << std::endl;

    std::cout << "------------------------------- Dataset Testing -------------------------------" << std::endl;

    Dataset dataset(csv, 5, {0}, 1, false);
    std::cout << dataset << std::endl;
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
    
    std::cout << "Destructor Calls" << std::endl;

    return 0;
}