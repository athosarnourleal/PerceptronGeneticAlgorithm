#ifndef GENALGNEURALNETWORK_UPLOADDATA_H
#define GENALGNEURALNETWORK_UPLOADDATA_H

#include <fstream>

#include "genalgNeuralNetwork.h"

inline void uploadGenerationScores(const element* bestFromEachGeneration, const int epochNum) {
    std::ofstream file("graphVisualizer/scores.txt", std::ios::app);

    for (int i = 0; i < epochNum; i++) {
        file << bestFromEachGeneration[i].score << '\n';
    }

    file.close();
}

inline void runVisualizer() {
    system("cd graphVisualizer");
    system("bash runVisualizer.sh");
}


#endif //GENALGNEURALNETWORK_UPLOADDATA_H
