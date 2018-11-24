//
// Created by egorpyat on 24.11.18.
//

#ifndef ATOM_CONFIGPARSER_H
#define ATOM_CONFIGPARSER_H

#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <sstream>

class Properties {
private:
    std::map<std::string, std::string> properties;

public:
    void load(std::ifstream &is);
    std::string getProperty(std::string key);

private:
    // String utils
    std::string trim(std::string data);
    std::string toLowerCase(std::string str);
    std::vector<std::string> split(std::string data, char delimiter);
};

#endif // ATOM_CONFIGPARSER_H
