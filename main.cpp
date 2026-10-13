#include <iostream>
#include "headers/genalgNeuralNetwork.h"
#include "headers/mnistLoading.h"
#include "headers/uploadData.h"
#include <omp.h>
#include <math.h>

static const DataSet *trainingData;
static NeuralNetwork *brain_buffer;

static int elementCounter = 0; // DEBUG
static double totalScore, iterationError;

static void runAndEvaluate(element *individual) {

    if (elementCounter % 100 == 0) {
        std::cout << "loading and running new element: " << elementCounter << '\n'; // DEBUG
    }
    elementCounter++;

    loadBrainFromDNA(brain_buffer, individual->dna);

    totalScore = 0;

    for (int trainingIteration = 0; trainingIteration < TRAINING_BATCH_SIZE; trainingIteration++) {
        runNeuralNetwork(trainingData->trainingBatchImages[trainingIteration], brain_buffer);
        iterationError = 0;

        for (int neuron = 0; neuron < OUTPUT_SIZE; neuron++) {
            if (neuron == trainingData->trainingBatchLabels[trainingIteration]) {
                // right answer --> expects: 1 --> error = (output - 1)²
                iterationError += brain_buffer->output[neuron] * (brain_buffer->output[neuron] - 2) + 1;
            } else {
                // wrong answer --> expects: 0 --> error = (output)²
                iterationError += brain_buffer->output[neuron] * brain_buffer->output[neuron];
            }

            totalScore += OUTPUT_SIZE / (iterationError + OUTPUT_SIZE);
        }
    }

    individual->score = totalScore;
}

constexpr int EPOCHS = 40;

int main() {
    rng.seed(42);

    NUM_THREADS = omp_get_max_threads();

    trainingData = new DataSet;

    brain_buffer = new NeuralNetwork;
    element* bestFromEachGeneration = new element[EPOCHS];
    dna* lastGenerationBuffer = new dna[POPULATION_SIZE];
    element* curGeneration = new element[POPULATION_SIZE];
    createPopulation(curGeneration);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        elementCounter = 0;

        // get batch
        std::cout << "gathering training batch" << '\n'; // DEBUG
        trainingData->assembleTrainingBatch();

        // execute population
        std::cout << "executing population code" << '\n'; // DEBUG
        for (int j = 0; j < POPULATION_SIZE; j++) {
            runAndEvaluate(curGeneration);
        }

        // generate next population
        std::cout << "generate next gen" << '\n'; // DEBUG
        generationElitism(curGeneration, lastGenerationBuffer);

        // get best from last generation
        const int bestElementIndex = findBest(curGeneration);
        memcpy(&bestFromEachGeneration[epoch].dna, &lastGenerationBuffer[bestElementIndex],GENE_NUMBER); // save best DNA
        bestFromEachGeneration[epoch].score = curGeneration[bestElementIndex].score; // save best score

        std::cout << std::endl << "generation " << epoch << ": " << bestFromEachGeneration[epoch].score  << '\n';
        if (epoch > 0) {
            std::cout << "difference: " << (bestFromEachGeneration[epoch].score - bestFromEachGeneration[epoch-1].score)  << '\n';
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

    std::cout << '\n' << "first score: " << bestFromEachGeneration[0].score << '\n';
    std::cout << "bestScore: " << bestFromEachGeneration[allTimeBestIndex].score << '\n';

    // save for graph visualization

    uploadGenerationScores(bestFromEachGeneration,EPOCHS);

    // save all time best in file

    saveBestIndividual(bestFromEachGeneration[allTimeBestIndex].dna, GENE_NUMBER);

    // deallocate all used heap memory

    delete trainingData;
    delete [] curGeneration;
    delete brain_buffer;
    delete [] lastGenerationBuffer;
    delete [] bestFromEachGeneration;

    // run visualization

    system("bash runVisualizer.sh");

    return 0;
}
