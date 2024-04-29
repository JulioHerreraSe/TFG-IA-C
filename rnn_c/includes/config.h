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

#include "../includes/RNN.h"
#include "../includes/dataReader.h"


#define INPUT_SIZE 9  // Ajusta este valor al número correcto de entradas (Features) de tu dataset
#define HIDDEN_SIZE 8  // Decide el tamaño de tu capa oculta
#define OUTPUT_SIZE 1  // Para regresión, suele ser 1
#define LEARNING_RATE 0.0006  // Ajusta este valor según sea necesario
#define EPOCHS 1 // Número de epocas de entrenamiento
#define MAX_LINE_LENGTH 1024


#endif //MLP_UPDATE_CONFIG_H
