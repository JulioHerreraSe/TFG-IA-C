#include "../headers/Layer_Dense.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Inicializa el generador de números aleatorios
void initializeRandom() {
    srand(time(NULL));
}

// Genera un número aleatorio entre -1 y 1
double randomWeight() {
    return (double)rand() / RAND_MAX * 2.0 - 1.0;
}

// Inicializa los pesos y bias de la capa densa con valores aleatorios
void initializeDenseLayer(Dense_Layer *layer) {
    for (int i = 0; i < NUM_INPUTS; i++) {
        for (int j = 0; j < NUM_OUTPUTS; j++) {
            layer->weights[i][j] = randomWeight();
        }
    }
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        layer->bias[i] = randomWeight();
    }
}

// Función para aplicar la capa densa a un conjunto de entradas
void applyDenseLayer(Dense_Layer *layer, double input[NUM_INPUTS], double output[NUM_OUTPUTS]) {
    for (int j = 0; j < NUM_OUTPUTS; j++) {
        output[j] = layer->bias[j]; // Inicializa con el bias
        for (int i = 0; i < NUM_INPUTS; i++) {
            output[j] += input[i] * layer->weights[i][j]; // Suma el producto de las entradas y los pesos
        }
    }
}
