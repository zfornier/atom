CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
PROJECT(confparser_lib VERSION 0.1
        DESCRIPTION "Configuration parser library"
        LANGUAGES CXX)

set(SOURCES Properties.cpp)

ADD_LIBRARY(confparser_lib ${SOURCES})

