#ifndef GENALGNEURALNETWORK_UPLOADDATA_H
#define GENALGNEURALNETWORK_UPLOADDATA_H

#include <filesystem>
#include <fstream>

#include "genalgNeuralNetwork.h"

inline void uploadGenerationScores(const element* bestFromEachGeneration, const int epochNum) {
    std::ofstream fileWriter("graphVisualizer/scores.txt", std::ios::trunc);

    for (int i = 0; i < epochNum; i++) {
        fileWriter << bestFromEachGeneration[i].score << '\n';
    }

    fileWriter.close();
}

inline void saveBestIndividual(const gene* bestIndividualDNA, const int geneNumber) {
    std::ofstream fileWriter("graphVisualizer/best.csv", std::ios::trunc);

    for (int i = 0; i < geneNumber; i++) {
        fileWriter << static_cast<unsigned int>(bestIndividualDNA[i].data) << ',';
    }

    fileWriter.close();
}


#endif //GENALGNEURALNETWORK_UPLOADDATA_H
