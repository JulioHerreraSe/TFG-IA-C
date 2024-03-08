#ifndef RNN_H
#define RNN_H

// Incluyendo el archivo de configuración para obtener las definiciones de tamaño
#include "config.h"

// Estructura para una capa de RNN
typedef struct {
    double *state; // Estado actual de la RNN
    double **weights_in; // Pesos para entradas a la RNN
    double **weights_state; // Pesos para el estado recurrente
    double **weights_out; // Pesos para la salida de la RNN
    double *bias; // Sesgos para las neuronas en la capa recurrente
    int inputSize; // Tamaño de la entrada
    int stateSize; // Tamaño del estado recurrente
    int outputSize; // Tamaño de la salida
} RNNLayer;

// Estructura para almacenar los gradientes de la RNN
typedef struct {
    double **gradWeights_in; // Gradientes para pesos de entrada
    double **gradWeights_state; // Gradientes para pesos del estado
    double *gradBias; // Gradientes para los sesgos
    double **gradWeights_out; // Gradientes para pesos de salida
} RNNLayerGradients;

// Declaraciones de funciones para operar con la RNN
void initializeRNNLayer(RNNLayer *layer, int inputSize, int stateSize, int outputSize);
void forwardRNNLayer(RNNLayer *layer, const double *input, double *output);
void calculateGradients(RNNLayer *layer, RNNLayerGradients *grads, const double *input, const double *outputError);
void updateRNNLayerWeights(RNNLayer *layer, RNNLayerGradients *grads, double learningRate);
void freeRNNLayer(RNNLayer *layer);

#endif // RNN_H
