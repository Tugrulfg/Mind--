#include "../include/Mind++.hpp"
#include <iostream>
#include <vector>

using namespace cmind;

int main(){
    // Shape shape({10,10});
    Tensor<float> tensor1({10});
    Tensor<float> tensor2({10, 10});

    // std::cout << "------------------------------- Shape Testing -------------------------------" << std::endl;
    // std::cout << tensor[0].shape << std::endl;
    // std::cout << tensor[0] << std::endl;


    std::cout << "------------------------------- Tensor Testing -------------------------------" << std::endl;
    std::cout << "Tensor1: \n" << tensor2[0] << std::endl;

    std::cout << "Assign1" << std::endl;
    for(int i=0; i<10; i++)
        tensor1[i] = i;

    tensor2[0] = tensor1;
    std::cout << "Assign2" << std::endl;

    std::cout << "Tensor2: \n" << tensor1 << std::endl;


    return 0;
}