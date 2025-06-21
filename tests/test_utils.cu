#include "test_utils.h"

#include <fstream>
#include <sstream>

void read_csv(const std::string& filepath, 
              std::vector<float>& data,
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
            data.push_back(std::stof(value));
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
