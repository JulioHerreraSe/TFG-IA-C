// mlp.h
#ifndef MLP_H
#define MLP_H

#include "../includes/config.h"

typedef struct {
    double **weights;
    double *weightsState;
    double *bias;
    double *state;
    double *lastState;
    int inputSize;
    int outputSize;
} Layer;

typedef struct {
    Layer hiddenLayer;
    Layer outputLayer;
    double lastError;
} RNN;

void initializeRNN(RNN *rnn, int inputSize, int hiddenSize, int outputSize);
double forward(RNN *rnn, double *input, double *hiddenOutput);
double trainMSE(RNN *rnn, double *input, double target, double learningRate);
void testRNN(const RNN *rnn, double **X_test, double *y_test, int numSamplesTest, FILE *file);
void guardarPesosYBiasCSV(const RNN *rnn, const char* archivoPesosCapaOculta, const char* archivoBiasCapaOculta,
                               const char* archivoPesosStateCapaOculta, const char* archivoPesosCapaSalida,
                               const char* archivoBiasCapaSalida, const char* archivoPesosStateCapaSalida);
void cargarPesosYBiasCSV(RNN *rnn, const char* archivoPesosCapaOculta, const char* archivoBiasCapaOculta,
                         const char* archivoPesosStateCapaOculta, const char* archivoPesosCapaSalida,
                         const char* archivoBiasCapaSalida, const char* archivoPesosStateCapaSalida);
void freeRNN(RNN *rnn);



#endif
