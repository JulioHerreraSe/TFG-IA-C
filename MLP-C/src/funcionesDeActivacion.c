//
// Created by julio on 19/12/2023.
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


void lineal(int TamCapa,  double *entrada, double *salida) {
    int i;
    for (i = 0; i < TamCapa; ++i) {
        salida[i] = entrada[i];
    }
}

void linealDerivada(int TamCapa, double *derivada) {
    int i;
    for (i = 0; i < TamCapa; ++i) {
        derivada[i] = 1;
    }
}

void tanhActivacion(int TamCapa, double *entrada, double *salida) {
    int i;
    for (i = 0; i < TamCapa; ++i) {
        salida[i] = tanh(entrada[i]);
    }
}

void tanhDerivada(int TamCapa, double *entrada, double *derivada) {
    double aux;
    int i;
    for (i = 0; i < TamCapa; ++i) {
        aux = tanh(entrada[i]);
        derivada[i] = (1 + aux) * (1 - aux);
    }
}

void sigmoid(int TamCapa, double *entrada, double *salida) {
    int i;
    for (i = 0; i < TamCapa; ++i) {
        salida[i] = 1 / (1 + exp(-entrada[i]));
    }
}

void sigmoidDerivada(int TamCapa, double *entrada, double *derivada) {
    int i;
    for (i = 0; i < TamCapa; ++i) {
        derivada[i] = entrada[i] * (1 - entrada[i]);
    }
}

void relu(int TamCapa, double *entrada, double *salida) {
    int i;
    for (i = 0; i < TamCapa; ++i) {
        salida[i] = (entrada[i] > 0.0) ? entrada[i] : 0.0;
    }
}

void reluDerivada(int TamCapa, double *entrada, double *derivada) {
    int i;
    for (i = 0; i < TamCapa; i++) {
        derivada[i] = (entrada[i] > 0.0) ? 1 : 0.0;
    }
}

void activacion(int funcion, int tamCapa, double *entrada, double *salida){
    switch (funcion) {
        case 1: // identity
            lineal(tamCapa, entrada, salida);
            break;
        case 2: // sigmoid
            sigmoid(tamCapa, entrada,salida);
            break;
        case 3: // tanh
            tanhActivacion(tamCapa,entrada, salida);
            break;
        case 4: // relu
            relu(tamCapa, entrada, salida);
            break;
        default:
            printf("Función incorrceta.\n");
            exit(0);
            break;
    }
}
