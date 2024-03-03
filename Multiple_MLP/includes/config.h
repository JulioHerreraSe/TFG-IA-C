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

#include "mlp.h"
#include "utils.h"

#define INPUT_SIZE 11 // Número de características
#define OUTPUT_SIZE 1 // Solo una salida, la calidad del vino
#define HIDDEN_LAYERS 2 // Por ejemplo, dos capas ocultas
#define HIDDEN_LAYER_SIZE 4
#define LEARNING_RATE 0.001
#define EPOCHS 1000
#define MAX_LINE_LENGTH 1024


#endif //MLP_UPDATE_CONFIG_H
