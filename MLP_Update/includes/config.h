//
// Created by julio on 20/02/2024.
//

#ifndef MLP_UPDATE_CONFIG_H
#define MLP_UPDATE_CONFIG_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <windows.h>
#include <ntsecapi.h>

#include "../includes/mlp.h"
#include "../includes/dataReader.h"


#define INPUT_SIZE 11  // Ajusta este valor al número correcto de entradas de tu dataset
#define HIDDEN_SIZE 3  // Decide el tamaño de tu capa oculta
#define OUTPUT_SIZE 1  // Para regresión, suele ser 1
#define LEARNING_RATE 0.005  // Ajusta este valor según sea necesario
#define EPOCHS 10000 // Número de epocas de entrenamiento
#define MAX_LINE_LENGTH 1024
#define NUM_FEATURES 11  // Ajusta este valor al número de características en tu dataset



#endif //MLP_UPDATE_CONFIG_H
