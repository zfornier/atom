//
// Created by egorpyat on 24.11.18.
//

#include "../../../include/ConfigParser/Properties.h"

void Properties::load(std::ifstream &is) {
    std::string line;
    std::vector<std::string> values;
    int lineNum = 0;
    while (std::getline(is, line)) {
        ++lineNum;

        std::size_t commentIndex = line.find('#');

        if(commentIndex != std::string::npos) {
            line.erase(commentIndex);
        }

        line = this->trim(line);

        if(!line.empty()) {
            values = this->split(this->toLowerCase(line), '=');
            if(values.size() == 2) {
                this->properties.insert({values[0], values[1]});
            }
            else {
                this->properties.insert({values[0], ""});
            }
        }
        else {
            continue;
        }
    }
}

std::string Properties::getProperty(std::string key) {
    return this->properties[this->toLowerCase(key)];
}

// String utils

std::string Properties::toLowerCase(std::string data) {
    std::transform(data.begin(), data.end(), data.begin(), ::tolower);
    return data;
}

std::vector<std::string> Properties::split(std::string data, char delimiter) {
    std::stringstream ss(data);
    std::string token;
    std::vector<std::string> tokens;

    while(std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

std::string Properties::trim(std::string str) {
    str.erase(std::remove_if(str.begin(), str.end(), isspace), str.end());
    return str;
}

double Properties::stringToDouble(std::string str) {
    try {
        return std::stod(str);
    }
    catch (std::invalid_argument &e) {
        throw std::invalid_argument("Error at parsing value: " + str);
    }
    catch (std::out_of_range &e) {
        throw std::out_of_range("Error at parsing value: " + str);
    }

}

int Properties::stringToInt(std::string str) {
    try {
        return std::stoi(str);
    }
    catch (std::invalid_argument &e) {
        throw std::invalid_argument("invalid argument: " + str);
    }
    catch (std::out_of_range &e) {
        throw std::out_of_range("out of range: " + str);
    }
}