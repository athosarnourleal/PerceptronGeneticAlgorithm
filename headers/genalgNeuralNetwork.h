#ifndef GENALGNEURALNETWORK_H
#define GENALGNEURALNETWORK_H

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

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

// random number generating // TODO: update random engine
inline int randomInt() {
    return rand();
}
inline int randomInt(const int &range) {
    return rand() % range;
}
constexpr float INVERSE_RAND_MAX = 4.65661e-10;
inline float randomDouble() {
    return rand() * INVERSE_RAND_MAX;
}

// --------------------------------------------------------------------------------------------------------------------- NEURAL NETWORK

constexpr int LAYER_SIZES[] = {1, 1};
constexpr int LAYER_NUMBER = 2;

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

inline float runBuffer[BIGGEST_LAYER_SIZE * 2];

struct NeuralNetwork {
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

inline float quickDotProduct(const float* vector1, const  float* vector2, const int size) {
    if (size % 2 == 0) return dotProductUnroll2(vector1, vector2, size);

    return dotProductUnroll2(vector1, vector2, size-1) + (vector1[size-1]*vector2[size-1]);
}

inline float ReLU(const float u) {
    return (u > 0) * u; // calculates "(u > 0) ? u : -0";
}

inline void runNeuralNetwork(const float *inputs, NeuralNetwork *brain) {
    memcpy(runBuffer, inputs, LAYER_SIZES[0] * sizeof(float)); // load input into buffer

    int weightPointer = 0, biasPointer = 0;
    int curBufferStart = BIGGEST_LAYER_SIZE, lastBufferStart = 0; // selects which half of the buffer is used

    // run
    for (int layer = 1; layer < LAYER_NUMBER; layer++) {
        for (int neuron = 0; neuron < LAYER_SIZES[layer]; neuron++) {
            runBuffer[curBufferStart + neuron]
               = brain->bias[biasPointer] + quickDotProduct(&runBuffer[lastBufferStart], &brain->weights[weightPointer], LAYER_SIZES[layer-1]);

            if (layer < LAYER_NUMBER-1) {
                runBuffer[curBufferStart + neuron] = ReLU(runBuffer[curBufferStart + neuron]);
            }

            weightPointer += LAYER_SIZES[layer-1]; // go to the set of weights of the next neuron
        }
        biasPointer++;
        swap(curBufferStart, lastBufferStart);
    }

    memcpy(brain->output, &runBuffer[lastBufferStart], LAYER_SIZES[LAYER_NUMBER-1] * sizeof(float)); // load results into output
}


// --------------------------------------------------------------------------------------------------------------------- GENALG


// DNA MANIPULATION //

constexpr int GENE_NUMBER = WEIGHT_NUMBER + BIAS_NUMBER; // number of genes

constexpr int GENE_DATA_BITS = 8;

constexpr float MIN_WEIGHT_VAL = -4, MAX_WEIGHT_VAL = 4; // values for translation
constexpr float MIN_BIAS_VAL = -6, MAX_BIAS_VAL = 6; // values for translation

constexpr int SIZEOF_GENE_STRUCT = 1; // 1 byte
struct gene { // 1 byte
    unsigned int data : GENE_DATA_BITS;
    int : 8 - GENE_DATA_BITS; // padding
};

constexpr int GENE_CAPACITY = powCompileTime(2, GENE_DATA_BITS);

constexpr float CONVERSION_DENOMINATOR = 1.0f / (GENE_CAPACITY - 1.0f);
inline float decodeGeneValue(const gene gene, const float min, const float max) {
    return min + static_cast<float>(gene.data) * (max - min) * CONVERSION_DENOMINATOR;
}

inline void loadBrainFromDNA(NeuralNetwork *brain, const gene* dna) {

    // std::cout << std::endl << "loading...: " << std::endl;
    int i;
    for (i = 0; i < WEIGHT_NUMBER; i++) { // load into weights
        brain->weights[i] = decodeGeneValue(*dna, MIN_WEIGHT_VAL, MAX_WEIGHT_VAL);
        dna++;
        // std::cout << "weights: " << brain->weights[i] << std::endl;
    }
    for (i = 0; i < BIAS_NUMBER; i++) { // load into weights
        brain->bias[i] = decodeGeneValue(*dna, MIN_BIAS_VAL, MAX_BIAS_VAL);
        dna++;
        // std::cout << "bias: " << brain->bias[i] << std::endl;
    }
}

// POPULATION MANIPULATING //

constexpr int POPULATION_SIZE = 2000;

struct element {
    gene dna[GENE_NUMBER];
    float score;
};

inline void createIndividual(element* individual) {
    for (int i = 0; i < GENE_NUMBER; i++) {
        individual->dna[i].data = randomInt(GENE_CAPACITY);
    }
    individual->score = static_cast<float>(RAND_MAX);
}

inline void createPopulation(element* population) {

    for (int i = 0; i < POPULATION_SIZE; i++) {
        createIndividual(&population[i]);
    }
}

// EVOLUTION //

inline int roulette(const element* population, const float scoreSum) {
    const float threshold = randomDouble() * scoreSum;

    float aux = population[0].score;
    int index = 0;
    while (aux < threshold) {
        aux += population[index++].score;
    }

    return index;
}


// create buffer in cache memory --- TODO: check if it isn't too big for cache when applying Neural Networks to .minst
static element childDnaBuffer;

inline void crossover(const gene* parent1, const gene* parent2) {
    const int cut = randomInt(GENE_NUMBER);

    for (int i = 0; i < GENE_NUMBER; i++) {
        childDnaBuffer.dna[i].data = (i < cut) ? parent1[i].data : parent2[i].data;
    }
}

constexpr float mutationChance = 0.1; // 0% - 100%
inline void mutate() {
    for (int i = 0; i < GENE_NUMBER; i++) {

        for (int j = 0; j < GENE_DATA_BITS; j++) {
            if (randomDouble() < mutationChance) {
                childDnaBuffer.dna[i].data ^= 0x01 << j; // flip bit
            }
        }
    }
}


inline void generation(element* curGeneration, gene* lastGenerationBuffer) {

    float scoreSum = 0;
    for (int i = 0; i < POPULATION_SIZE; i++) {
        memcpy(&lastGenerationBuffer[i * GENE_NUMBER], curGeneration[i].dna, GENE_NUMBER * sizeof(gene));

        scoreSum += curGeneration[i].score;
    }

    for (int i = 0; i < POPULATION_SIZE; i++) {
        crossover(
            &lastGenerationBuffer[roulette(curGeneration, scoreSum) * GENE_NUMBER], // select parent1
            &lastGenerationBuffer[roulette(curGeneration, scoreSum) * GENE_NUMBER]  // select parent2
        );

        mutate();

        memcpy(curGeneration[i].dna, childDnaBuffer.dna, GENE_NUMBER * sizeof(gene)); // save result from buffer to the next generation
    }

    // OBS: the scores are NOT meant to be reset in generation(), since they will be overriden when running the neural network again
}

constexpr int ELITE_SIZE = POPULATION_SIZE*0.1;
static int elite[ELITE_SIZE];

inline void generationElitism(element* curGeneration, gene* lastGenerationBuffer) {

    for (int i = 0; i < ELITE_SIZE; i++) {
        elite[i] = i;
    }

    static float scoreSum = 0;
    for (int i = 0; i < POPULATION_SIZE; i++) {
        memcpy(&lastGenerationBuffer[i * GENE_NUMBER], curGeneration[i].dna, GENE_NUMBER * sizeof(gene));

        scoreSum += curGeneration[i].score;

        int j;
        for (j = ELITE_SIZE-1; j >= 0; j--) {
            if (curGeneration[i].score < curGeneration[elite[j]].score) {
                j++;
                break;
            }
        }
        if (j < ELITE_SIZE) {
            elite[j] = i;
        }
    }

    static bool inElite = false;
    for (int i = 0; i < POPULATION_SIZE; i++) {
        inElite = false;
        for (int j = 0; j < ELITE_SIZE; j++) {
            if (i == elite[j]) inElite = true;
        }
        if (inElite) continue;


        crossover(
            &lastGenerationBuffer[roulette(curGeneration, scoreSum) * GENE_NUMBER], // select parent1
            &lastGenerationBuffer[roulette(curGeneration, scoreSum) * GENE_NUMBER]  // select parent2
        );

        mutate();

        memcpy(curGeneration[i].dna, childDnaBuffer.dna, GENE_NUMBER * sizeof(gene)); // save result from buffer to the next generation
    }

    // OBS: the scores are NOT meant to be reset in generation(), since they will be overriden when running the neural network again
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

inline int findBestMinimizeError(const element* pop) {
    int best = 0;
    for (int i = 1; i < POPULATION_SIZE; i++) {
        if (pop[i].score < pop[best].score) {
            best = i;
        }
    }

    return best;
}

#endif // GENALGNEURALNETWORK_H
