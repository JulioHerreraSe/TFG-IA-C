//
// Created by julio on 01/03/2024.
//

#ifndef MULTIPLE_MLP_UTILS_H
#define MULTIPLE_MLP_UTILS_H

#include "config.h"

// Funciones de activación y sus derivadas
double sigmoid(double x);
double sigmoidDerivative(double x);
void readData(const char* filePath, double*** features, double** target, int* numSamples);

#endif //MULTIPLE_MLP_UTILS_H
