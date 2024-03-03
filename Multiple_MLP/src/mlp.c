#include "../includes/config.h"

// Asumiendo que las definiciones de las estructuras NeuralNetwork y Layer están en mlp.h
// Asumiendo también que utils.h proporciona las funciones necesarias como sigmoid y sigmoidDerivative

void initializeNetwork(NeuralNetwork *nn) {
    int layerSizes[HIDDEN_LAYERS + 2] = {INPUT_SIZE}; // Comenzar con el tamaño de la capa de entrada
    // Definir los tamaños de las capas ocultas y la capa de salida
    for (int i = 1; i <= HIDDEN_LAYERS; i++) {
        layerSizes[i] = HIDDEN_LAYER_SIZE;
    }
    layerSizes[HIDDEN_LAYERS + 1] = OUTPUT_SIZE; // Definir el tamaño de la capa de salida

    nn->numLayers = sizeof(layerSizes) / sizeof(layerSizes[0]);
    nn->layers = (Layer *)malloc(nn->numLayers * sizeof(Layer));

    for (int i = 0; i < nn->numLayers; ++i) {
        nn->layers[i].inputSize = i == 0 ? INPUT_SIZE : layerSizes[i - 1];
        nn->layers[i].outputSize = layerSizes[i];
        nn->layers[i].weights = (double **)malloc(nn->layers[i].outputSize * sizeof(double *));
        nn->layers[i].bias = (double *)malloc(nn->layers[i].outputSize * sizeof(double));
        nn->layers[i].output = (double *)malloc(nn->layers[i].outputSize * sizeof(double));

        for (int j = 0; j < nn->layers[i].outputSize; ++j) {
            nn->layers[i].weights[j] = (double *)malloc(nn->layers[i].inputSize * sizeof(double));
            nn->layers[i].bias[j] = 0; // Inicializar bias a 0
            for (int k = 0; k < nn->layers[i].inputSize; ++k) {
                nn->layers[i].weights[j][k] = ((double)rand() / (double)RAND_MAX) * 0.2 - 0.1; // Inicialización aleatoria
            }
        }
    }
}

// Implementar forward, backpropagation y freeNetwork aquí

void forward(NeuralNetwork *nn, const double *inputs, double *output) {
    double *currentActivation = malloc(nn->layers[0].inputSize * sizeof(double));
    memcpy(currentActivation, inputs, nn->layers[0].inputSize * sizeof(double));

    for (int i = 0; i < nn->numLayers; ++i) {
        Layer *layer = &nn->layers[i];
        double *nextActivation = malloc(layer->outputSize * sizeof(double));

        for (int j = 0; j < layer->outputSize; ++j) {
            double activationSum = layer->bias[j];
            for (int k = 0; k < layer->inputSize; ++k) {
                activationSum += currentActivation[k] * layer->weights[j][k];
            }
            nextActivation[j] = (i == nn->numLayers - 1) ? activationSum : sigmoid(activationSum); // No activation on the last layer for regression
        }

        if (i > 0) free(currentActivation); // Liberar la activación anterior, excepto la entrada
        currentActivation = nextActivation; // La siguiente activación se convierte en la actual
    }

    memcpy(output, currentActivation, nn->layers[nn->numLayers - 1].outputSize * sizeof(double)); // Copiar la última activación a la salida
    free(currentActivation); // Liberar la última activación
}

void backpropagation(NeuralNetwork *nn, const double *inputs, const double *targets, double learningRate) {
    double *outputs = malloc(nn->layers[nn->numLayers - 1].outputSize * sizeof(double));
    forward(nn, inputs, outputs); // Primero, realiza un forward pass para obtener las salidas

    // Inicializar delta para la capa de salida
    double *delta = malloc(nn->layers[nn->numLayers - 1].outputSize * sizeof(double));
    for (int i = 0; i < nn->layers[nn->numLayers - 1].outputSize; ++i) {
        delta[i] = targets[i] - outputs[i]; // Error de la capa de salida
    }

    // Propagar el error hacia atrás y actualizar pesos y sesgos
    for (int i = nn->numLayers - 1; i >= 0; --i) {
        Layer *layer = &nn->layers[i];
        double *inputToLayer = i == 0 ? (double *)inputs : nn->layers[i - 1].output;

        for (int j = 0; j < layer->outputSize; ++j) {
            for (int k = 0; k < layer->inputSize; ++k) {
                double error = delta[j];
                if (i != nn->numLayers - 1) { // Para capas ocultas, ajustar el error por la derivada de la sigmoide
                    error *= sigmoidDerivative(layer->output[j]);
                }
                layer->weights[j][k] += learningRate * error * inputToLayer[k];
            }
            layer->bias[j] += learningRate * delta[j];
        }

        // Calcular el delta para la siguiente capa hacia atrás
        if (i > 0) {
            double *newDelta = malloc(nn->layers[i - 1].outputSize * sizeof(double));
            for (int j = 0; j < nn->layers[i - 1].outputSize; ++j) {
                newDelta[j] = 0;
                for (int k = 0; k < layer->outputSize; ++k) {
                    newDelta[j] += delta[k] * layer->weights[k][j];
                }
                if (i > 1) { // Ajustar por la derivada de la sigmoide para capas ocultas
                    newDelta[j] *= sigmoidDerivative(nn->layers[i - 1].output[j]);
                }
            }
            free(delta);
            delta = newDelta;
        }
    }

    free(delta); // Liberar el último delta
    free(outputs); // Liberar las salidas calculadas
}

void freeNetwork(NeuralNetwork *nn) {
    for (int i = 0; i < nn->numLayers; ++i) {
        Layer *layer = &nn->layers[i];
        for (int j = 0; j < layer->outputSize; ++j) {
            free(layer->weights[j]);
        }
        free(layer->weights);
        free(layer->bias);
    }
    free(nn->layers);
}

