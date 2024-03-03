#ifndef MLP_H
#define MLP_H

#include "config.h" // Asegúrate de incluir config.h para obtener las constantes de configuración

typedef struct {
    int inputSize, outputSize;
    double **weights, *bias;
    double *output; // Salidas después de aplicar la activación
} Layer;

typedef struct {
    Layer *layers;
    int numLayers;
} NeuralNetwork;

void initializeNetwork(NeuralNetwork *nn);
void forward(NeuralNetwork *nn, const double *input, double *output);
void backpropagation(NeuralNetwork *nn, const double *inputs, const double *targets, double learningRate);
void freeNetwork(NeuralNetwork *nn);

#endif
