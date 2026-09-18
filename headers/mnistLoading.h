#ifndef GENALGNEURALNETWORK_MNISTLOADING_H
#define GENALGNEURALNETWORK_MNISTLOADING_H
#include <algorithm>
#include <assert.h>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>

#include "genalgNeuralNetwork.h"

inline std::string trainingDataFileName = "mnist_train.csv";

constexpr int TRAINING_BATCH_SIZE = 800;

class DataSet {
public:
    int *trainingBatchLabels;
    double **trainingBatchImages;

    explicit DataSet() {
        std::cout << "loading training data..." << '\n';

        getDataDimensions();

        dataSetLabels = new int[dataSetSize];
        dataSetImages = new double[dataSetSize*imageSize];

        trainingBatchLabels = new int[TRAINING_BATCH_SIZE];
        trainingBatchImages = new double*[TRAINING_BATCH_SIZE];

        loadData();

        std::cout << "data successfully loaded." << '\n';
    }

    void assembleTrainingBatch() const {
        for (int i = 0; i < TRAINING_BATCH_SIZE; i++) {
            const int chosen = randomInt(TRAINING_BATCH_SIZE);
            trainingBatchLabels[i] = dataSetLabels[chosen]; // store label
            trainingBatchImages[i] = &dataSetImages[chosen*imageSize]; // store reference to image
        }
    }

    ~DataSet() {
        delete [] trainingBatchLabels;
        delete [] trainingBatchImages;

        delete [] dataSetLabels;
        delete [] dataSetImages;
    }

private:
    int *dataSetLabels;
    double *dataSetImages;

    std::size_t dataSetSize = 0;
    std::size_t imageSize = 0;

    void loadData() const {
        std::ifstream file(trainingDataFileName);
        if (file.is_open() == false) {
            std::cerr << "file não foi encontrada ou não pode ser aberta" << '\n';
            exit(404);
        }

        std::string line;

        getline(file, line); // skip header

        // load and parse files
        int curLine = 0;
        while (getline(file, line)) {
            int argStart = 0, imageVectorPosition = 0;;
            for (int i = 0; i < line.length(); i++) {
                if (line[i] == ',' || i+1 == line.length()) {
                    if (argStart == 0) {
                        dataSetLabels[curLine] = stoi(line.substr(argStart, i - argStart));
                    } else {
                        dataSetImages[curLine*imageSize + imageVectorPosition] = stof(line.substr(argStart, i - argStart));
                        imageVectorPosition++;
                    }

                    argStart = i + 1;
                }
            }

            curLine++;
        }
    }

    void getDataDimensions() {
        std::ifstream file(trainingDataFileName);
        if (file.is_open() == false) {
            std::cerr << "Arquivo não foi encontrado ou não pode ser aberto" << '\n';
            exit(404);
        }

        std::string line;

        getline(file, line); // get header

        imageSize = std::count(line.begin(), line.end(), ',');

        dataSetSize = 0;
        while (getline(file, line)) {
            dataSetSize++;
        }
    }
};

#endif //GENALGNEURALNETWORK_MNISTLOADING_H
