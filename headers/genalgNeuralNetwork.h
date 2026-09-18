#ifndef GENALGNEURALNETWORK_H
#define GENALGNEURALNETWORK_H

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

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

// random number generating // TODO: update random engine later
inline int randomInt() {
    return rand();
}
inline int randomInt(const int &range) {
    return rand() % range;
}
constexpr double INVERSE_RAND_MAX = 4.65661e-10;
inline double randomDouble() {
    return rand() * INVERSE_RAND_MAX;
}

// --------------------------------------------------------------------------------------------------------------------- NEURAL NETWORK

constexpr int LAYER_SIZES[] = {28*28, 16, 16, 10};
constexpr int LAYER_NUMBER = 4;
constexpr int OUTPUT_SIZE = LAYER_SIZES[LAYER_NUMBER-1];

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

static double runBuffer[BIGGEST_LAYER_SIZE * 2];

struct NeuralNetwork {
    double weights[WEIGHT_NUMBER];
    double bias[BIAS_NUMBER];
    double output[LAYER_SIZES[LAYER_NUMBER-1]];
};

inline double dotProductUnroll2(const double* vector1, const  double* vector2, const int size) {
    double sum1 = 0.0;// uses two cores automatically(in theory...)
    double sum2 = 0.0;

    for (int i = 0; i < size; i+=2) {
        sum1 += vector1[i] * vector2[i];
        sum2 += vector1[i+1] * vector2[i+1];
    }

    return sum1 + sum2;
}

inline double quickDotProduct(const double* vector1, const  double* vector2, const int size) {
    if (size % 2 == 0) return dotProductUnroll2(vector1, vector2, size);

    return dotProductUnroll2(vector1, vector2, size-1) + vector1[size-1]*vector2[size-1];
}

inline double ReLU(const double u) {
    return (u > 0) * u; // calculates "(u > 0) ? u : -0";
}

inline void runNeuralNetwork(const double *inputs, NeuralNetwork *brain) {
    memcpy(runBuffer, inputs, LAYER_SIZES[0] * sizeof(double)); // load input into buffer

    int weightPointer = 0;
    int curBufferStart = BIGGEST_LAYER_SIZE, lastBufferStart = 0; // selects which half of the buffer is used

    double outputMin = 0, outputMax = 0;
    // run
    for (int layer = 1; layer < LAYER_NUMBER; layer++) {
        for (int neuron = 0; neuron < LAYER_SIZES[layer]; neuron++) {

            if (layer < LAYER_NUMBER-1) {
                // apply ReLU in the hidden layers only
                runBuffer[curBufferStart + neuron]
                   = ReLU(brain->bias[layer - 1] + quickDotProduct(&runBuffer[lastBufferStart], &brain->weights[weightPointer], LAYER_SIZES[layer-1]));

            } else {
                // last layer
                runBuffer[curBufferStart + neuron]
                   = brain->bias[layer - 1] + quickDotProduct(&runBuffer[lastBufferStart], &brain->weights[weightPointer], LAYER_SIZES[layer-1]);

                if (runBuffer[curBufferStart + neuron] < outputMin) {
                    outputMin = runBuffer[curBufferStart + neuron];
                }
                if (runBuffer[curBufferStart + neuron] > outputMax) {
                    outputMax = runBuffer[curBufferStart + neuron];
                }
            }

            weightPointer += LAYER_SIZES[layer-1]; // go to the set of weights of the next neuron
        }
        swap(curBufferStart, lastBufferStart);
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) { // normalize and load into output
        brain->output[i] = (runBuffer[lastBufferStart + i] - outputMin) / (outputMax - outputMin);
    }
}

inline int getBiggestOutputIndex(const NeuralNetwork *brain) {
    int indexBiggest = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (brain->output[i] > brain->output[indexBiggest]) {
            indexBiggest = i;
        }
    }
    return indexBiggest;
}

// --------------------------------------------------------------------------------------------------------------------- GENALG


// GENETIC MANIPULATION //

constexpr int GENE_NUMBER = WEIGHT_NUMBER + BIAS_NUMBER; // number of genes

constexpr int GENE_DATA_BITS = 8; // {   0 < GENE_DATA_BITS < 9   }
constexpr int GENE_CAPACITY = powCompileTime(2, GENE_DATA_BITS);

constexpr double MIN_WEIGHT_VAL = -10, MAX_WEIGHT_VAL = 10; // values for translation
constexpr double MIN_BIAS_VAL = -10, MAX_BIAS_VAL = 10; // values for translation

struct gene {
    unsigned char data : GENE_DATA_BITS;
    unsigned char : 8 - GENE_DATA_BITS; // padding
};

constexpr double CONVERSION_DENOMINATOR = 1.0f / (GENE_CAPACITY - 1.0f);
inline double decodeGeneValue(const gene gene, const double min, const double max) {
    return min + static_cast<double>(gene.data) * (max - min) * CONVERSION_DENOMINATOR;
}

inline void loadBrainFromDNA(NeuralNetwork *brain, const gene* dna) {
    int dnaCounter = 0;
    for (int i = 0; i < WEIGHT_NUMBER; i++) { // load into weights
        brain->weights[i] = decodeGeneValue(dna[dnaCounter++], MIN_WEIGHT_VAL, MAX_WEIGHT_VAL);
    }
    for (int i = 0; i < BIAS_NUMBER; i++) { // load into weights
        brain->bias[i] = decodeGeneValue(dna[dnaCounter++], MIN_BIAS_VAL, MAX_BIAS_VAL);
    }
}

// POPULATION MANIPULATING //

constexpr int POPULATION_SIZE = 1000;

struct element {
    gene dna[GENE_NUMBER];
    double score;
};

inline void createIndividual(element* individual) {
    for (int i = 0; i < GENE_NUMBER; i++) {
        individual->dna[i].data = randomInt(GENE_CAPACITY);
    }
    individual->score = 0;
}

inline void createPopulation(element* population) {
    for (int i = 0; i < POPULATION_SIZE; i++) {
        createIndividual(&population[i]);
    }
}

// EVOLUTION //

inline int roulette(const element *population, const double scoreSum) {
    const double threshold = randomDouble() * scoreSum;

    double aux = population[0].score;

    int index = 0;
    while (aux <= threshold) {
        if (index >= POPULATION_SIZE) {
            std::cout << "deu erro!" << '\n';
            throw std::runtime_error("deu erro!");
            break;
        }
        aux += population[index].score;
        index++;
    }

    return index;
}

// create buffer in cache memory --- TODO: check if it isn't too big for cache when applying this code to the .minst
static element childDnaBuffer;

inline void crossover(const gene* parent1, const gene* parent2) {
    const int cut = randomInt(GENE_NUMBER*GENE_DATA_BITS + 1);

    const int cutGene = cut / GENE_DATA_BITS;
    const int cutBit = cut% GENE_DATA_BITS;
    const int cutBitFromRight = GENE_DATA_BITS - cutBit;

    for (int i = 0; i < GENE_NUMBER; i++) {
        if (i < cutGene) {
            childDnaBuffer.dna[i] = parent1[i];
        } else if (i > cutGene) {
            childDnaBuffer.dna[i] = parent2[i];
        } else {
            // merge both genes
            childDnaBuffer.dna[i].data =
                ((parent2[i].data >> cutBitFromRight) << cutBitFromRight) | (parent2[i].data >> cutBit);
        }
    }
}

constexpr double MUTATION_CHANCE = 0.25; // 0% - 100%
inline void mutation() {
    for (int i = 0; i < GENE_NUMBER; i++) {
        for (int j = 0; j < GENE_DATA_BITS; j++) {
            if (randomDouble() < MUTATION_CHANCE) {
                childDnaBuffer.dna[i].data ^= 0b1 << j; // flip bit
            }
        }
    }
}

struct dna {
    gene genes[GENE_NUMBER];
};

inline void generation(element* curGeneration, dna* lastGenerationBuffer) {
    double scoreSum = 0;
    for (int i = 0; i < POPULATION_SIZE; i++) {
        memcpy(&lastGenerationBuffer[i], curGeneration[i].dna, GENE_NUMBER * sizeof(gene));

        scoreSum += curGeneration[i].score;
    }

    for (int i = 0; i < POPULATION_SIZE; i++) {
        crossover(
            lastGenerationBuffer[roulette(curGeneration, scoreSum)].genes, // select parent1
            lastGenerationBuffer[roulette(curGeneration, scoreSum)].genes  // select parent2
        );

        mutation();

        memcpy(curGeneration[i].dna, childDnaBuffer.dna, GENE_NUMBER * sizeof(gene)); // save result from buffer to the next generation
    }
}

constexpr int ELITE_SIZE = POPULATION_SIZE*0.1;
inline void generationElitism(element* curGeneration, dna* lastGenerationBuffer) {
    double scoreSum = 0;
    int elite[ELITE_SIZE];

    for (int i = 0; i < ELITE_SIZE; i++) {
        elite[i] = i % POPULATION_SIZE;
    }

    for (int i = 0; i < POPULATION_SIZE; i++) {
        memcpy(&lastGenerationBuffer[i], curGeneration[i].dna, GENE_NUMBER * sizeof(gene));

        scoreSum += curGeneration[i].score;

        if (curGeneration[i].score > curGeneration[elite[ELITE_SIZE - 1]].score) {
            int j = ELITE_SIZE-1;
            while (j >= 0 && curGeneration[elite[j]].score < curGeneration[i].score) {
                j--;
            }
            j++;

            if (i == elite[j]) continue;

            for (int k = j+1; k < ELITE_SIZE; k++) {
                elite[k] = elite[k - 1];
            }
            elite[j] = i;
        }
    }

    static bool isElite = false;
    for (int i = 0; i < POPULATION_SIZE; i++) {

        isElite = false;
        for (int j = 0; j < ELITE_SIZE; j++) {
            if (elite[j] == i) {
                isElite = true;
                break;
            }
        }
        if (isElite) continue;

        crossover(
            lastGenerationBuffer[roulette(curGeneration, scoreSum)].genes, // select parent1
            lastGenerationBuffer[roulette(curGeneration, scoreSum)].genes  // select parent2
        );

        mutation();

        memcpy(curGeneration[i].dna, childDnaBuffer.dna, GENE_NUMBER * sizeof(gene)); // save result from buffer to the next generation
    }
}

// EVALUATION //

inline int findBest(const element* pop) {
    int best = 0;
    for (int i = 1; i < POPULATION_SIZE; i++) {
        if (pop[best].score < pop[i].score) {
            best = i;
        }
    }
    return best;
}

#endif // GENALGNEURALNETWORK_H