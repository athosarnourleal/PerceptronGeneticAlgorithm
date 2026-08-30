#ifndef GENALGNEURALNETWORK_H
#define GENALGNEURALNETWORK_H

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
// #include "randomPool.h"


// --------------------------------------------------------------------------------------------------------------------- UTIL

constexpr int absCompileTime(const int n) {
    if (n < 0) return -n;

    return n;
}

constexpr int powRecursion(const int a, const int n) {
    if (n == 1) return a;
    const int aux = powRecursion(a, n/2);
    return aux*aux;
}
constexpr int powCompileTime(const int a, const int n) {
    if (n % 2 == 0) {
        return powRecursion(a, absCompileTime(n));
    }
    return powRecursion(a, absCompileTime(n)-1) * a;
}

inline void swap(int &x, int &y) noexcept {
    x = y ^ x;
    y = x ^ y;
    x = y ^ x;
}

// --------------------------------------------------------------------------------------------------------------------- NEURAL NETWORK


constexpr int LAYER_SIZES[] = {1, 6, 6, 1};
constexpr int LAYER_NUMBER = 4;

constexpr int calculateWeightNumber() {
    int weightNum = 0;

    for (int i = 1; i < LAYER_NUMBER; i++) {
        weightNum += LAYER_SIZES[i]*LAYER_SIZES[i-1];
    }

    return weightNum;
}

constexpr int getBiggestLayerSize() {
    int size = 0;

    for (int i = 0; i < LAYER_NUMBER; i++) {
        if (LAYER_SIZES[i] > size)  {
            size = LAYER_SIZES[i];
        }
    }

    return size;
}

constexpr int BIGGEST_LAYER_SIZE = getBiggestLayerSize();

constexpr int WEIGHT_NUMBER = calculateWeightNumber(), BIAS_NUMBER = LAYER_NUMBER-1;

inline float brainExecutionBuffer[BIGGEST_LAYER_SIZE * 2];

struct NeuralNetwork {// size = (WEIGHT_NUMBER+BIAS_NUMBER+LAYER_SIZES[LAYER_NUMBER-1])*__SIZEOF_FLOAT__ Bytes
    float weights[WEIGHT_NUMBER];
    float bias[BIAS_NUMBER];
    float output[LAYER_SIZES[LAYER_NUMBER-1]];
};

inline float dotProductUnroll2(const float* vector1, const  float* vector2, const int size) {
    float sum1 = 0.0;// uses two cores automatically(in theory)
    float sum2 = 0.0;

    for (int i = 0; i < size; i+=2) {
        sum1 += vector1[i] * vector2[i];
        sum2 += vector1[i+1] * vector2[i+1];
    }

    return sum1 + sum2;
}

inline float quick_dotProduct(const float* vector1, const  float* vector2, const int size) {
    if (size % 2 == 0) return dotProductUnroll2(vector1, vector2, size);

    return dotProductUnroll2(vector1, vector2, size-1) + (vector1[size-1]*vector2[size-1]);
}

inline float ReLU(const float u) {
    return (u > 0) * u; // calculates "(u > 0) ? u : -0";
}

inline void runNeuralNetwork(const float* inputs,NeuralNetwork *brain) {

    memcpy(brainExecutionBuffer, inputs, LAYER_SIZES[0] * sizeof(float)); // load input into buffer

    int weightPointer = 0, biasPointer = 0;
    int curBufferStart = BIGGEST_LAYER_SIZE, lastBufferStart = 0; // selects which half of the buffer is used

    for (int layer = 1; layer < LAYER_NUMBER; layer++) {
        for (int neuron = 0; neuron < LAYER_SIZES[layer]; neuron++) {
            brainExecutionBuffer[curBufferStart + neuron]
                = ReLU(brain->bias[biasPointer] + quick_dotProduct(&brainExecutionBuffer[lastBufferStart], &brain->weights[weightPointer], LAYER_SIZES[layer-1]));

            weightPointer += LAYER_SIZES[layer-1]; // go to the set of weights of the next neuron
        }
        biasPointer++;
        swap(curBufferStart, lastBufferStart);
    }

    memcpy(brain->output, &brainExecutionBuffer[lastBufferStart], LAYER_SIZES[LAYER_NUMBER-1] * sizeof(float)); // load results into output

    std::cout << weightPointer << std::endl;
}


// --------------------------------------------------------------------------------------------------------------------- GENALG


// DNA MANIPULATION //

constexpr int GENE_NUMBER = WEIGHT_NUMBER + BIAS_NUMBER; // number of genes

constexpr int GENE_DATA_BITS = 6;
constexpr int GENE_DATA_CAPACITY = 6;
constexpr float MIN_VAL = -4, MAX_VAL = 4;

constexpr int SIZEOF_GENE_STRUCT = 1; // 1 byte
struct gene { // 1 byte
    int data : GENE_DATA_BITS;
    int : 8 - GENE_DATA_BITS; // padding
};


constexpr int GENE_CAPACITY = powCompileTime(2, GENE_DATA_BITS);
constexpr float INV_GENE_CAPACITY = 1.0f / (  GENE_CAPACITY - 1.0f  );
inline float decodeGeneValue(const struct gene gene) {
    return MIN_VAL + static_cast<float>(gene.data) * (MAX_VAL - MIN_VAL) * INV_GENE_CAPACITY;
}

inline void loadBrainFromDNA(NeuralNetwork *brain, const gene* dna) {
    int i;

    for (i = 0; i < WEIGHT_NUMBER; i++) { // load into weights
        brain->weights[i] = decodeGeneValue(*dna++);
    }
    for (i = 0; i < BIAS_NUMBER; i++) { // load into weights
        brain->bias[i] = decodeGeneValue(*dna++);
    }
}

// POPULATION MANIPULATING //

constexpr int POPULATION_SIZE = 100;

struct element {
    gene dna[GENE_NUMBER];
};

inline void createIndividual(element* individual) {
    for (int i = 0; i < GENE_NUMBER; i++) {
        individual->dna[i].data = rand() % GENE_CAPACITY;
    }
}

inline element* createPopulation() {
    element* population = new element[POPULATION_SIZE];

    for (int i = 0; i < POPULATION_SIZE; i++) {
        createIndividual(&population[i]);
    }

    return population;
}

// EVOLUTION //

// TODO: evolutionary operators + evolutionary function


#endif // GENALGNEURALNETWORK_H
