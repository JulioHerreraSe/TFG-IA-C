#ifndef CONFIG_H
#define CONFIG_H

#include <stdio.h>

// Tamaño de la entrada, depende de las características de tu serie temporal
#define INPUT_SIZE 4

// Tamaño de la capa oculta, puede ser ajustado según la complejidad necesaria
#define HIDDEN_SIZE 5

// Tamaño de la salida, para regresión de series temporales suele ser 1
#define OUTPUT_SIZE 1

// Tasa de aprendizaje, ajustar según sea necesario para la convergencia
#define LEARNING_RATE 0.001

// Número de épocas para el entrenamiento
#define EPOCHS 1000

// Tamaño del batch para el entrenamiento, importante para el SGD
#define BATCH_SIZE 32

// Longitud de la secuencia para las RNN, ajustar según la memoria temporal necesaria
#define SEQUENCE_LENGTH 20

#endif // CONFIG_H
