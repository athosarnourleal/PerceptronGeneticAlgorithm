#include <iostream>
#include "headers/genalgNeuralNetwork.h"
#include "headers/uploadData.h"

using namespace std;

// BASIC TESTING EXAMPLE
constexpr float T_W = 2, T_B = 4;

static float getSolution(const int x) {
    return T_W * static_cast<float>(x) + T_B;
}


// TODO: create library for Dataset Management
constexpr int BATCH_SIZE = 2000;

static float* trainingBatch = new float[BATCH_SIZE*2];
constexpr int training_range = 3000;

static float* createTrainingBatch(float *batch) {
    for (int i = 0; i < BATCH_SIZE; i++) {
        batch[i*2] = randomDouble()*training_range*2 - training_range;
        if (randomDouble() >= 0.5) {
            batch[i*2] *= -1;
        }

        batch[i*2+1] = getSolution(batch[i*2]);
    }

    return batch;
}

constexpr float INV_BATCH_SIZE = 1.0f / static_cast<float>(BATCH_SIZE);
static void evaluate(NeuralNetwork *brain, element *individual) {
    float avgError = 0;
    for (int i = 0; i < BATCH_SIZE; ++i) {
        // run
        runNeuralNetwork(&trainingBatch[i*2], brain);

        avgError += abs(trainingBatch[i*2 + 1] - brain->output[0]);
    }

    avgError = avgError * INV_BATCH_SIZE;
    individual->score = abs(1 / (avgError + 1));
}

constexpr int EPOCHS = 1000;

int main() {
    srand(42);

    element* bestFromEachGeneration = new element[EPOCHS * GENE_NUMBER];

    dna* lastGenerationBuffer = new dna[POPULATION_SIZE];
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
            evaluate(&brains[j], &curGeneration[j]);
        }

        // generate next population
        generation(curGeneration, lastGenerationBuffer);

        // get best from last generation
        const int bestElementIndex = findBest(curGeneration);
        memcpy(&bestFromEachGeneration[epoch].dna, &lastGenerationBuffer[bestElementIndex],GENE_NUMBER); // save best DNA
        bestFromEachGeneration[epoch].score = curGeneration[bestElementIndex].score; // save best score

        if (epoch % 10 == 0) {
            std::cout << endl << "generation " << epoch << ": " << bestFromEachGeneration[epoch].score  << endl;
            if (epoch > 0) {
                std::cout << "difference: " << (bestFromEachGeneration[epoch].score - bestFromEachGeneration[epoch-1].score) << endl;
            }
        }
    }

    // get all time best

    int allTimeBestIndex = 0;
    for (int i = 1; i < EPOCHS; i++) {
        if (bestFromEachGeneration[i].score > bestFromEachGeneration[allTimeBestIndex].score) {
            allTimeBestIndex = i;
        }
    }

    // RESULTS

    cout << endl << "first score: " << bestFromEachGeneration[0].score << endl;
    cout << "bestScore: " << bestFromEachGeneration[allTimeBestIndex].score << endl;

    loadBrainFromDNA(&brains[0], bestFromEachGeneration[allTimeBestIndex].dna);

    cout << "weight: " << brains[0].weights[0] << endl;
    cout << "bias: " << brains[0].bias[0] << endl << endl;

    // save for graph visualization

    uploadGenerationScores(bestFromEachGeneration,EPOCHS);

    // save all time best in file

    saveBestIndividual(bestFromEachGeneration[allTimeBestIndex].dna, GENE_NUMBER);

    // deallocate all used heap memory
    delete [] curGeneration;
    delete [] brains;
    delete [] lastGenerationBuffer;
    delete [] trainingBatch;
    delete [] bestFromEachGeneration;

    // system("bash runVisualizer.sh");

    return 0;
}