#include <iostream>
#include "headers/genalgNeuralNetwork.h"
#include "headers/uploadData.h"

using namespace std;

// BASIC TESTING EXAMPLE
constexpr float T_W = 2, T_B = 4;

static float getSolution(const int x) {
    return T_W * static_cast<float>(x) + T_B;
}

constexpr int BATCH_SIZE = 2000;

// TODO: create library for Dataset Management
static float* trainingBatch = new float[BATCH_SIZE*2];

constexpr int training_range = 100;
static float* createTrainingBatch(float *batch) {
    for (int i = 0; i < BATCH_SIZE; i++) {
        batch[i*2] = randomDouble() * training_range * 2 - training_range;
        batch[i*2+1] = getSolution(batch[i]);
    }

    return batch;
}

constexpr float errorMax = 1000000;

static void runByBatch(NeuralNetwork *brain, element *individual) {
    float avgError = 0;

    for (int i = 0; i < BATCH_SIZE; ++i) {
        // run
        runNeuralNetwork(&trainingBatch[i*2], brain);

        // calculate absolute error
        avgError += abs(trainingBatch[i*2+1] - brain->output[0]);

    }
    avgError /= BATCH_SIZE;

    // cout << "avgError: " << avgError << endl;
    individual->score = (avgError >= errorMax) ? 0.0f : errorMax - avgError;
    
    // cout << "score: " << individual->score << endl;
}



constexpr int EPOCHS = 4000;

int main() {
    /*
    srand(time(nullptr));

    element* bestFromEachGeneration = new element[EPOCHS * GENE_NUMBER];

    gene* lastGenerationBuffer = new gene[GENE_NUMBER * POPULATION_SIZE];
    element* curGeneration = new element[POPULATION_SIZE];
    createPopulation(curGeneration);

    NeuralNetwork* brains = new NeuralNetwork[POPULATION_SIZE];

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // load brain
        for (int j = 0; j < POPULATION_SIZE; j++) {
            loadBrainFromDNA(&brains[j], curGeneration[j].dna);
        }

        // get batch
        createTrainingBatch(trainingBatch);

        // execute population
        for (int j = 0; j < POPULATION_SIZE; j++) {
            runByBatch(&brains[j], &curGeneration[j]);
        }

        // generate next population
        generationElitism(curGeneration, lastGenerationBuffer);

        // get best from last generation
        const int bestElementIndex = findBest(curGeneration);
        memcpy(&bestFromEachGeneration[epoch].dna, &lastGenerationBuffer[bestElementIndex],GENE_NUMBER); // save best DNA
        bestFromEachGeneration[epoch].score = curGeneration[bestElementIndex].score; // save best score

        std::cout << endl << "generation " << epoch << ": " << bestFromEachGeneration[epoch].score  << endl;

        if (epoch >= 1) {
            std::cout << "difference: " << (bestFromEachGeneration[epoch].score - bestFromEachGeneration[epoch-1].score) << endl;
        }
    }

    // get all time best

    int allTimeBestIndex = 0;
    for (int i = 1; i < EPOCHS; i++) {
        if (bestFromEachGeneration[i].score > bestFromEachGeneration[allTimeBestIndex].score) {
            allTimeBestIndex = i;
        }
    }

    // DEBUG

    cout << endl << "first score: " << bestFromEachGeneration[0].score << endl;
    cout << "bestScore: " << bestFromEachGeneration[allTimeBestIndex].score << endl;
    loadBrainFromDNA(&brains[0], bestFromEachGeneration[allTimeBestIndex].dna);

    cout << "weight: " << brains[0].weights[0] << endl;
    cout << "bias: " << brains[0].bias[0] << endl;

    // save for graph visualization

    uploadGenerationScores(bestFromEachGeneration,EPOCHS);

    // save all time best in file


    // deallocate
    delete [] curGeneration;
    delete [] brains;
    delete [] lastGenerationBuffer;
    delete [] trainingBatch;
    */

    runVisualizer();

    return 0;
}

