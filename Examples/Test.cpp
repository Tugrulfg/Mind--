#include "../include/Mind++.hpp"
#include <iostream>
#include <vector>


int main(){
    Shape shape({10,10});
    Tensor<float> tensor({10, 10});

    std::cout << "------------------------------- Shape Testing -------------------------------" << std::endl;
    std::cout << tensor.shape << std::endl;
    std::cout << tensor.shape.size() << std::endl;



    // for(const int& dim: tensor.shape)
    //     std::cout << dim << " ";
    // std::cout << std::endl;
    return 0;
}