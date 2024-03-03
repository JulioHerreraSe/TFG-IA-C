// mlp.h
#ifndef MLP_H
#define MLP_H

#include "../includes/config.h"

typedef struct {
    double **weights;
    double *bias;
    int inputSize;
    int outputSize;
} Layer;

typedef struct {
    Layer hiddenLayer;
    Layer outputLayer;
} MLP;

void initializeMLP(MLP *mlp, int inputSize, int hiddenSize, int outputSize);
void train(MLP *mlp, double *input, double target, double learningRate);
double forward(MLP *mlp, double *input, double *hiddenOutput);
void testMLP(const MLP *mlp, double **X_test, double *y_test, int numSamplesTest);
void guardarPesosYBiasCSV(const MLP *mlp, const char* archivoPesosCapaOculta, const char* archivoBiasCapaOculta, const char* archivoPesosCapaSalida, const char* archivoBiasCapaSalida);
    void cargarPesosYBiasCSV(MLP *mlp, const char* archivoPesosCapaOculta, const char* archivoBiasCapaOculta, const char* archivoPesosCapaSalida, const char* archivoBiasCapaSalida);
void freeMLP(MLP *mlp);



#endif
