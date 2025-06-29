#ifndef TEST_UTILS_IMPL_H
#define TEST_UTILS_IMPL_H

#include "test_utils.h"

#include <fstream>
#include <sstream>

template <typename T>
void read_csv(const std::string& filepath, 
              std::vector<T>& data,
              int& rows,
              int& cols) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }
    data.clear();
    std::string line;
    rows = 0;
    cols = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        int current_cols = 0;
        while (std::getline(ss, value, ' ')) {
            if constexpr (std::is_same<T, float>::value) {
                data.push_back(std::stof(value));
            }
            else if constexpr (std::is_same<T, int>::value) {
                data.push_back(std::stoi(value));
            }
            else {
                static_assert(always_false<T>::value, "Unsupported type for CSV reading");
            }
            current_cols++;
        }
        if (rows == 0) {
            cols = current_cols; // Set the number of columns from the first row
        } else if (current_cols != cols) {
            throw std::runtime_error("Inconsistent number of columns in CSV file.");
        }
        rows++;
    }
}

template <typename T>
T read_scalar(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }
    T value;
    file >> value;
    if (file.fail()) {
        throw std::runtime_error("Failed to read scalar value from file: " + filepath);
    }
    return value;
}

#endif // TEST_UTILS_IMPL_H
