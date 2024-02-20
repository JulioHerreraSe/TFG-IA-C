// mlp.h
#ifndef MLP_H
#define MLP_H

#include <stdlib.h>
#include <math.h>
#include <time.h>

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
void freeMLP(MLP *mlp);

#endif
