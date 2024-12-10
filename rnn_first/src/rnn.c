#include "../includes/rnn.h"
#include <stdlib.h>
#include <math.h>

// Función de activación y su derivada
double tanhActivation(double x) {
    return tanh(x);
}

double derivativeTanh(double x) {
    return 1.0 - x * x;
}

// Inicialización de la capa RNN
void initializeRNNLayer(RNNLayer *layer, int inputSize, int stateSize, int outputSize) {
    layer->inputSize = inputSize;
    layer->stateSize = stateSize;
    layer->outputSize = outputSize;

    // Asignación dinámica de memoria para los pesos y el estado
    layer->state = (double *)calloc(stateSize, sizeof(double));
    layer->weights_in = (double **)malloc(stateSize * sizeof(double *));
    layer->weights_state = (double **)malloc(stateSize * sizeof(double *));
    layer->weights_out = (double **)malloc(outputSize * sizeof(double *));
    layer->bias = (double *)calloc(stateSize, sizeof(double));

    // Inicializar los pesos de entrada, estado y salida
    for (int i = 0; i < stateSize; i++) {
        layer->weights_in[i] = (double *)malloc(inputSize * sizeof(double));
        layer->weights_state[i] = (double *)malloc(stateSize * sizeof(double));
        for (int j = 0; j < inputSize; j++) {
            layer->weights_in[i][j] = ((double)rand() / RAND_MAX) * 2 - 1; // Peso inicial aleatorio
        }
        for (int j = 0; j < stateSize; j++) {
            layer->weights_state[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }
    for (int i = 0; i < outputSize; i++) {
        layer->weights_out[i] = (double *)malloc(stateSize * sizeof(double));
        for (int j = 0; j < stateSize; j++) {
            layer->weights_out[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }
}

// Propagación hacia adelante
void forwardRNNLayer(RNNLayer *layer, const double *input, double *output) {
    double *newState = (double *)calloc(layer->stateSize, sizeof(double));

    for (int i = 0; i < layer->stateSize; i++) {
        newState[i] = layer->bias[i];
        for (int j = 0; j < layer->inputSize; j++) {
            newState[i] += layer->weights_in[i][j] * input[j];
        }
        for (int j = 0; j < layer->stateSize; j++) {
            newState[i] += layer->weights_state[i][j] * layer->state[j];
        }
        newState[i] = tanhActivation(newState[i]);
    }

    // Calcula la salida
    for (int i = 0; i < layer->outputSize; i++) {
        output[i] = 0;
        for (int j = 0; j < layer->stateSize; j++) {
            output[i] += layer->weights_out[i][j] * newState[j];
        }
    }

    // Actualiza el estado de la RNN
    for (int i = 0; i < layer->stateSize; i++) {
        layer->state[i] = newState[i];
    }
    free(newState);
}
void calculateGradients(RNNLayer *layer, RNNLayerGradients *grads, const double *input, const double *outputError) {
    // Imagina que outputError es el error calculado entre la salida predicha y la real.
    // Este método simplificado calcula los gradientes para un paso de tiempo.
    // Nota: En una implementación real de BPTT, acumularías gradientes a lo largo de varios pasos de tiempo.

    for (int i = 0; i < layer->stateSize; i++) {
        double derivative = derivativeTanh(layer->state[i]);

        for (int j = 0; j < layer->inputSize; j++) {
            grads->gradWeights_in[i][j] += derivative * input[j] * (*outputError);
        }
        for (int j = 0; j < layer->stateSize; j++) {
            grads->gradWeights_state[i][j] += derivative * layer->state[j] * (*outputError);
        }
        grads->gradBias[i] += derivative * (*outputError);
    }

    for (int i = 0; i < layer->outputSize; i++) {
        for (int j = 0; j < layer->stateSize; j++) {
            grads->gradWeights_out[i][j] += layer->state[j] * (*outputError);
        }
    }
}

void updateRNNLayerWeights(RNNLayer *layer, RNNLayerGradients *grads, double learningRate) {
    // Actualiza los pesos y el bias con los gradientes calculados y la tasa de aprendizaje.
    for (int i = 0; i < layer->stateSize; i++) {
        for (int j = 0; j < layer->inputSize; j++) {
            layer->weights_in[i][j] -= learningRate * grads->gradWeights_in[i][j];
        }
        for (int j = 0; j < layer->stateSize; j++) {
            layer->weights_state[i][j] -= learningRate * grads->gradWeights_state[i][j];
        }
        layer->bias[i] -= learningRate * grads->gradBias[i];
    }

    for (int i = 0; i < layer->outputSize; i++) {
        for (int j = 0; j < layer->stateSize; j++) {
            layer->weights_out[i][j] -= learningRate * grads->gradWeights_out[i][j];
        }
    }
}

void freeRNNLayer(RNNLayer *layer) {
    // Liberar el estado de la RNN
    free(layer->state);

    // Liberar los pesos de las entradas
    for (int i = 0; i < layer->stateSize; i++) {
        free(layer->weights_in[i]);
    }
    free(layer->weights_in);

    // Liberar los pesos del estado
    for (int i = 0; i < layer->stateSize; i++) {
        free(layer->weights_state[i]);
    }
    free(layer->weights_state);

    // Liberar los pesos de las salidas
    for (int i = 0; i < layer->outputSize; i++) {
        free(layer->weights_out[i]);
    }
    free(layer->weights_out);

    // Liberar los sesgos
    free(layer->bias);.
}
