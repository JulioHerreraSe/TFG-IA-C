
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../includes/dataReader.h"

#define INPUT_SIZE 2
#define HIDDEN_SIZE 3
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.1

typedef struct {
    double weights[INPUT_SIZE][HIDDEN_SIZE];
    double bias[HIDDEN_SIZE];
} HiddenLayer;

typedef struct {
    double weights[HIDDEN_SIZE][OUTPUT_SIZE];
    double bias[OUTPUT_SIZE];
} OutputLayer;

typedef struct {
    HiddenLayer hiddenLayer;
    OutputLayer outputLayer;
} MLP;

double randRange(double min, double max) {
    return min + (rand() / (RAND_MAX / (max - min)));
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

void initializeMLP(MLP *mlp) {
    // Initialize hidden layer weights and biases
    for (int i = 0; i < INPUT_SIZE; ++i) {
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            mlp->hiddenLayer.weights[i][j] = randRange(-1.0, 1.0);
        }
    }
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        mlp->hiddenLayer.bias[i] = randRange(-1.0, 1.0);
    }

    // Initialize output layer weights and biases
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            mlp->outputLayer.weights[i][j] = randRange(-1.0, 1.0);
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        mlp->outputLayer.bias[i] = randRange(-1.0, 1.0);
    }
}

double forward(MLP *mlp, double input[], double *hiddenOutput, double *finalOutput) {
    // Forward pass through hidden layer
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        hiddenOutput[i] = 0.0;
        for (int j = 0; j < INPUT_SIZE; ++j) {
            hiddenOutput[i] += input[j] * mlp->hiddenLayer.weights[j][i];
        }
        hiddenOutput[i] += mlp->hiddenLayer.bias[i];
        hiddenOutput[i] = sigmoid(hiddenOutput[i]);
    }

    // Forward pass through output layer
    *finalOutput = 0.0;
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        *finalOutput += hiddenOutput[i] * mlp->outputLayer.weights[i][0];
    }
    *finalOutput += mlp->outputLayer.bias[0];
    // No activation function is applied in the output layer for regression

    return *finalOutput;
}

void train(MLP *mlp, double input[], double target, double learningRate) {
    double hiddenOutput[HIDDEN_SIZE];
    double finalOutput;
    double outputError, hiddenError[HIDDEN_SIZE];

    // Propagación hacia adelante para obtener la salida actual de la red
    forward(mlp, input, hiddenOutput, &finalOutput);

    // Calcula el error de la salida (Predicción - Real)
    outputError = finalOutput - target;

    // Backpropagation para la capa de salida
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        double gradOutputLayer = outputError * hiddenOutput[i];
        mlp->outputLayer.weights[i][0] -= learningRate * gradOutputLayer;
    }
    mlp->outputLayer.bias[0] -= learningRate * outputError;

    // Backpropagation para la capa oculta
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        hiddenError[i] = outputError * mlp->outputLayer.weights[i][0] * sigmoidDerivative(hiddenOutput[i]);

        for (int j = 0; j < INPUT_SIZE; ++j) {
            mlp->hiddenLayer.weights[j][i] -= learningRate * hiddenError[i] * input[j];
        }
        mlp->hiddenLayer.bias[i] -= learningRate * hiddenError[i];
    }
}


int main() {
    srand((unsigned int)time(NULL));
    MLP mlp;
    initializeMLP(&mlp);

    // Suponiendo que estos datos ya están normalizados
    double inputs[][INPUT_SIZE] = {{0.05, 0.1}, {0.1, 0.1}, {0.2, 0.25}, {0.15, 0.2}};
    double targets[] = {0.01, 0.02, 0.03, 0.025};

    // Definir índices para separar entrenamiento y prueba
    int training_indices[] = {0, 1, 2}; // Primeros 3 para entrenamiento
    int test_indices[] = {3}; // Último para prueba


    int epochs = 10000;
    double learningRate = LEARNING_RATE;
    double bestTestError = INFINITY;
    int epochsSinceImprovement = 0;
    const int patience = 1000; // Número de épocas para esperar antes de ajustar la tasa de aprendizaje

    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalTrainingError = 0.0;

        // Entrenamiento
        for (int i = 0; i < sizeof(training_indices)/sizeof(training_indices[0]); i++) {
            int idx = training_indices[i];
            double hiddenOutput[HIDDEN_SIZE], finalOutput;
            train(&mlp, inputs[idx], targets[idx], learningRate);
            forward(&mlp, inputs[idx], hiddenOutput, &finalOutput);
            double error = pow(finalOutput - targets[idx], 2);
            totalTrainingError += error;
        }

        // Prueba
        double totalTestError = 0.0;
        for (int i = 0; i < sizeof(test_indices)/sizeof(test_indices[0]); i++) {
            int idx = test_indices[i];
            double hiddenOutput[HIDDEN_SIZE], finalOutput;
            forward(&mlp, inputs[idx], hiddenOutput, &finalOutput);
            double error = pow(finalOutput - targets[idx], 2);
            totalTestError += error;

            if (epoch % 10 == 0) {
                printf("Training - Epoch %d, Sample %d, Prediction: %f, Target: %f, Error: %f\n", epoch, idx,
                       finalOutput, targets[idx], sqrt(error));
            }
        }

        // Ajuste de tasa de aprendizaje basado en el error de prueba
        if (totalTestError < bestTestError) {
            bestTestError = totalTestError;
            epochsSinceImprovement = 0;
        } else {
            epochsSinceImprovement++;
            if (epochsSinceImprovement > patience) {
                learningRate *= 0.9; // Reducir tasa de aprendizaje
                epochsSinceImprovement = 0;
                printf("Reduced learning rate to %f\n", learningRate);
            }
        }
    }
    read_csv();
    return 0;
}
