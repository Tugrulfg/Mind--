#include "../include/Math.hpp"
#include <cmath>

namespace cmind{

    // Return the absolute difference of the given tensors
    Tensor<float> abs_dif(const Tensor<float>& input1, const Tensor<float>& input2){
        if(input1.shape() != input2.shape()){
            std::cout << "abs_dif: Shapes do not match" << std::endl;
            abort();
        }
        Tensor<float> output(input1.shape());
        const float* in1 = input1.data();
        const float* in2 = input2.data();
        for(size_t i=0; i<input1.size(); i++){
            if(in1[i] < in2[i])
                output[i] = in2[i]-in1[i];
            else
                output[i] = in1[i]-in2[i];
        }
        return output;
    }

    // Square root of the given tensor
    Tensor<float> sqrt(const Tensor<float>& input){
        Tensor<float> output(input.shape());
        const float* in = input.data();

        for(size_t i=0; i<input.size(); i++)
            output[i] = std::sqrt(in[i]);

        return output;
    }





}