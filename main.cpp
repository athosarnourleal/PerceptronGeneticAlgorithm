#include <iostream>
#include "headers/genalgNeuralNetwork.h"

using namespace std;

// BASIC TESTING EXAMPLE
constexpr float T_M = 3, T_N = 4;

static float getSolution(const int x) {
    return T_M * static_cast<float>(x) + T_N;
}

constexpr int batchSize = 1000;

static float* trainingBatch = new float[batchSize*2];

static float* createTrainingBatch(float *batch) {
    for (int i = 0; i < batchSize*2; i+=2) {
        batch[i] = rand() % 100;
        batch[i+1] = getSolution(batch[i]);
    }

    return batch;
}

static void runByBatch(NeuralNetwork *brain, element *individual) {
    int score = 0;
    for (int i = 0; i < batchSize; ++i) {
        // run
        runNeuralNetwork(&trainingBatch[i], brain);

        // evaluate output
        if (brain->biggestOutput == trainingBatch[i+1]) {
            score++;
        }
    }

}

constexpr int EPOCHS = 10;

int main() {
    srand(time(nullptr));

    element* bestFromEachGeneration = new element[EPOCHS * GENE_NUMBER];

    gene* lastGenerationBuffer = new gene[GENE_NUMBER * POPULATION_SIZE];
    element* population = createPopulation();
    NeuralNetwork* brains = new NeuralNetwork[POPULATION_SIZE];

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // load brain
        for (int j = 0; j < POPULATION_SIZE; j++) {
            loadBrainFromDNA(&brains[j], population[j].dna);
        }

        // get batch
        createTrainingBatch(trainingBatch);

        // execute and evaluate population
        for (int j = 0; j < POPULATION_SIZE; j++) {
            runByBatch(&brains[j], &population[j]);
        }

        // generate next population
        

        // get best from last generation
        const int bestElementIndex = findBest(population);
        memcpy(&bestFromEachGeneration[epoch].dna, &lastGenerationBuffer[bestElementIndex],GENE_NUMBER); // save best's DNA
        bestFromEachGeneration[epoch].score = population[bestElementIndex].score; // save best's score
    }

    // get all time best


    // save all time best in file


    // deallocate
    delete [] population;
    delete [] brains;
    delete [] lastGenerationBuffer;
    delete [] trainingBatch;

    return 0;
}

