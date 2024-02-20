// dense_layer.h

#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <stdlib.h> // Incluir para size_t

#define NUM_INPUTS 3
#define NUM_OUTPUTS 2

// Estructura para la capa densa
typedef struct {
    double weights[NUM_INPUTS][NUM_OUTPUTS]; // Matriz de pesos
    double bias[NUM_OUTPUTS]; // Vector de bias
} Dense_Layer;

// Declara las funciones para ser usadas por otros archivos
void initializeRandom();
double randomWeight();
void initializeDenseLayer(Dense_Layer *layer);
void applyDenseLayer(Dense_Layer *layer, double input[NUM_INPUTS], double output[NUM_OUTPUTS]);

#endif // DENSE_LAYER_H
