
cmake_minimum_required(VERSION 4.3)

project(genAlgNeuralNetwork LANGUAGES CXX)

find_package(OpenMP COMPONENTS CXX REQUIRED)

set(CMAKE_CXX_STANDARD 20)

add_executable(
        genAlgNeuralNetwork
        main.cpp
        headers/genalgNeuralNetwork.h
        headers/mnistLoading.h
        headers/uploadData.h
)
target_link_libraries(genAlgNeuralNetwork PRIVATE OpenMP::OpenMP_CXX)
