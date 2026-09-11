#ifndef GENALGNEURALNETWORK_UPLOADDATA_H
#define GENALGNEURALNETWORK_UPLOADDATA_H

#include <filesystem>
#include <fstream>

#include "genalgNeuralNetwork.h"

inline void uploadGenerationScores(const element* bestFromEachGeneration, const int epochNum) {
    std::ofstream fileSopre("graphVisualizer/scores.txt", std::ios::trunc);

    for (int i = 0; i < epochNum; i++) {
        fileSopre << bestFromEachGeneration[i].score << '\n';
    }

    fileSopre.close();

    std::ofstream fileBrain("graphVisualizer/brain.txt", std::ios::trunc);

    NeuralNetwork *brain = new NeuralNetwork();

    for (int i = 0; i < epochNum; i++) {
        loadBrainFromDNA(brain, bestFromEachGeneration[i].dna);
        fileBrain << brain->weights[0] << '\n';
        fileBrain << brain->bias[0] << '\n';
    }

    fileBrain.close();

    delete brain;
}


#endif //GENALGNEURALNETWORK_UPLOADDATA_H
