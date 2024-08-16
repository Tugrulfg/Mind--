#include "../include/Utils.hpp"

namespace cmind{

    // Convert the input to polynomial features
    std::vector<float> poly_features(const std::vector<double>& input, int degree){
        std::vector<float> output;
        std::vector<std::vector<double>> polyFeatures = createPolynomialFeatures(input, degree);

        for (const auto& feature : polyFeatures) {
            output.push_back(feature[0]);
        }
        return output;
    }

    // Helper function to recursively generate polynomial features
    void generatePolynomialFeatures(const std::vector<double>& input, int degree, 
                                    int index, std::vector<int> powers, 
                                    std::vector<std::vector<int>>& allPowers) {
        if (index == input.size()) {
            int sum = 0;
            for (int power : powers) {
                sum += power;
            }
            if (sum > 0 && sum <= degree) {  // Skip the bias term (sum > 0)
                allPowers.push_back(powers);
            }
            return;
        }

        for (int i = 0; i <= degree; ++i) {
            powers[index] = i;
            generatePolynomialFeatures(input, degree, index + 1, powers, allPowers);
        }
    }

    std::vector<std::vector<double>> createPolynomialFeatures(const std::vector<double>& input, int degree) {
        std::vector<std::vector<int>> allPowers;
        std::vector<int> powers(input.size(), 0);
        
        // Generate all combinations of powers for each feature
        generatePolynomialFeatures(input, degree, 0, powers, allPowers);

        // Now convert these powers into actual polynomial features
        std::vector<std::vector<double>> polyFeatures;
        for (const auto& powerSet : allPowers) {
            double product = 1.0;
            for (int i = 0; i < input.size(); ++i) {
                product *= std::pow(input[i], powerSet[i]);
            }
            polyFeatures.push_back({product});
        }
        
        return polyFeatures;
    }

    // Splits the line into values according to the delimiter
    std::vector<std::string> split(const std::string& line, char delimiter) {
        std::vector<std::string> values;
        std::stringstream ss(line);
        std::string token;

        while (std::getline(ss, token, delimiter)) {
            token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());    // Trim spaces
            if(!token.empty())
                values.push_back(token);
        }

        return values;
    }
}