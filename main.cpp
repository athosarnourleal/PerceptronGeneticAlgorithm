#include <iostream>
#include "headers/genalgNeuralNetwork.h"
#include "headers/mnistLoading.h"
#include "headers/uploadData.h"

static const DataSet *trainingData;

static int elementCouter = 0; // DEBUG

static void runAndEvaluate(NeuralNetwork *brain, element *individual) {
    std::cout << "running new element: " << elementCouter++ << '\n'; // DEBUG

    double totalScore = 0;
    double iterationAvgError = 0;

    for (int i = 0; i < TRAINING_BATCH_SIZE; i++) {
        runNeuralNetwork(trainingData->trainingBatchImages[i], brain);

        for (int neuron = 0; neuron < OUTPUT_SIZE; neuron++) {
            if (neuron == trainingData->trainingBatchLabels[i]) {
                // right answer --> expects: 1
                iterationAvgError += (brain->output[neuron] - 1) * (brain->output[neuron] - 1);
            } else {
                // wrong answer --> expects: 0
                iterationAvgError += brain->output[neuron] * brain->output[neuron];
            }
        }

        totalScore += OUTPUT_SIZE / (iterationAvgError + OUTPUT_SIZE);
    }

    individual->score = totalScore;
}

constexpr int EPOCHS = 6;

int main() {
    srand(time(nullptr));

    trainingData = new DataSet;

    element* bestFromEachGeneration = new element[EPOCHS];

    NeuralNetwork* brains = new NeuralNetwork[POPULATION_SIZE];
    dna* lastGenerationBuffer = new dna[POPULATION_SIZE];
    element* curGeneration = new element[POPULATION_SIZE];
    createPopulation(curGeneration);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        elementCouter = 0;

        // get batch
        std::cout << "gathering training batch" << '\n'; // DEBUG
        trainingData->assembleTrainingBatch();

        // load brain
        std::cout << "decoding brains" << '\n'; // DEBUG
        for (int j = 0; j < POPULATION_SIZE; j++) {
            loadBrainFromDNA(&brains[j], curGeneration[j].dna);
        }

        // execute population
        std::cout << "executing population code" << '\n'; // DEBUG
        for (int j = 0; j < POPULATION_SIZE; j++) {
            runAndEvaluate(&brains[j], curGeneration);
        }

        // generate next population
        std::cout << "generate next gen" << '\n'; // DEBUG
        generation(curGeneration, lastGenerationBuffer);

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
    delete [] brains;
    delete [] lastGenerationBuffer;
    delete [] bestFromEachGeneration;

    // run visualization

    system("bash runVisualizer.sh");

    return 0;
}
