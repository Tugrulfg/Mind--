#include "../include/Utils.hpp"

namespace cmind{

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

    // Random number generator
    size_t random(size_t min, size_t max) {
        std::random_device rd;
        std::uniform_int_distribution<size_t> dist(min, max);
        return dist(rd);
    }
}