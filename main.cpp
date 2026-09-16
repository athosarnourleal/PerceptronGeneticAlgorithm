#include <iostream>
#include "headers/genalgNeuralNetwork.h"
#include "headers/mnistLoading.h"
#include "headers/uploadData.h"

static DataSet trainingData;

static double *inputs = new double {100.0};

static void evaluate(NeuralNetwork *brain, element *individual) {
    runNeuralNetwork(inputs, brain);

    // TODO: optimize later

    double totalScore = 0;
    for (int i = 0; i < TRAINING_BATCH_SIZE; i++) {

        double iterationAvgError = 0;
        for (int neuron = 0; neuron < OUTPUT_SIZE; neuron++) {
            double curError = 0;
            if (neuron == trainingData.trainingBatchLabels[i]) {
                // right answer --> expects: 1
                curError = brain->output[neuron] - 1;
            } else {
                // wrong answer --> expects: 0
                curError = brain->output[neuron] - 0;
            }

            iterationAvgError += curError*curError;
        }
        iterationAvgError /= OUTPUT_SIZE;

        totalScore += 1 / (iterationAvgError + 1);
    }

    individual->score = totalScore;
}

constexpr int EPOCHS = 200;

int main() {
    srand(42);

    element* bestFromEachGeneration = new element[EPOCHS];
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
        trainingData.assembleTrainingBatch();

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
                std::cout << std::endl << "generation " << epoch << ": " << bestFromEachGeneration[epoch].score  << '\n';
            if (epoch > 0) {
                std::cout << "difference: " << (bestFromEachGeneration[epoch].score - bestFromEachGeneration[epoch-1].score)  << '\n';
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

    std::cout << '\n' << "first score: " << bestFromEachGeneration[0].score << '\n';
    std::cout << "bestScore: " << bestFromEachGeneration[allTimeBestIndex].score << '\n';

    loadBrainFromDNA(&brains[0], bestFromEachGeneration[allTimeBestIndex].dna);

    runNeuralNetwork(inputs, &brains[0]);

    std::cout << "--- OUTPUTS: " << '\n';
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        std::cout << i << ": " << brains[0].output[i] << '\n';
    }
    std::cout << "biggestOutput: " << getBiggestOutputIndex(&brains[0]) << '\n';

    // save for graph visualization

    uploadGenerationScores(bestFromEachGeneration,EPOCHS);

    // save all time best in file

    saveBestIndividual(bestFromEachGeneration[allTimeBestIndex].dna, GENE_NUMBER);

    // deallocate all used heap memory

    delete inputs; // TODO: REMOVE DEBUGGING

    delete [] curGeneration;
    delete [] brains;
    delete [] lastGenerationBuffer;
    delete [] bestFromEachGeneration;

    // run visualization

    system("bash runVisualizer.sh");

    return 0;
}
