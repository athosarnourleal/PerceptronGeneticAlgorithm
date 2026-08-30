#include <iostream>

#include "headers/genalgNeuralNetwork.h"

using namespace std;

constexpr float T_M = 3, T_N = 4;

static float get_solution(const int x) {
    return T_M * x + T_N;
}


int main() {
    // BASIC TEST

    srand(time(nullptr));

    element* indv = new element;
    createIndividual(indv);

    NeuralNetwork *brain = new NeuralNetwork;
    loadBrainFromDNA(brain, indv->dna);

    // setup training env
    float* inputs = new float;
    *inputs = rand() % 1000;

    const float expectedOutput = get_solution(*inputs);

    // run
    runNeuralNetwork(inputs, brain);

    cout << "expected: " << expectedOutput << " --- recieved: " << brain->output[0] << endl;

    // deallocate
    delete inputs;
    delete indv;
    delete brain;
}

