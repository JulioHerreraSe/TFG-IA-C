#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

void train(MLP *mlp, double input[], double target) {
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
        mlp->outputLayer.weights[i][0] -= LEARNING_RATE * gradOutputLayer;
    }
    mlp->outputLayer.bias[0] -= LEARNING_RATE * outputError;

    // Backpropagation para la capa oculta
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        hiddenError[i] = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            hiddenError[i] += outputError * mlp->outputLayer.weights[i][j];
        }
        hiddenError[i] *= sigmoidDerivative(hiddenOutput[i]);

        for (int j = 0; j < INPUT_SIZE; ++j) {
            mlp->hiddenLayer.weights[j][i] -= LEARNING_RATE * hiddenError[i] * input[j];
        }
        mlp->hiddenLayer.bias[i] -= LEARNING_RATE * hiddenError[i];
    }
}



int main() {
    // Inicialización de la red
    MLP mlp;
    initializeMLP(&mlp);

    // Datos de entrenamiento: [Entradas] => [Target]
    double inputs[][INPUT_SIZE] = {{0.1, 0.2}, {0.2, 0.2}, {0.4, 0.5}};
    double targets[] = {0.3, 0.4, 0.7}; // Valores objetivos para regresión

    int epochs = 10000; // Número de épocas de entrenamiento

    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalError = 0.0;
        for (int i = 0; i < sizeof(targets)/sizeof(targets[0]); i++) {
            double hiddenOutput[HIDDEN_SIZE];
            double finalOutput;
            // Entrenamiento de la red con cada par de entrada-target
            train(&mlp, inputs[i], targets[i]);
            forward(&mlp, inputs[i], hiddenOutput, &finalOutput);

            // Calcular el error cuadrático para esta época
            double error = finalOutput - targets[i];
            totalError += error * error;

            // Imprimir los resultados cada 100 épocas
            if (epoch % 100 == 0) {
                printf("Epoch %d, Sample %d, Prediction: %f, Target: %f, Error: %f\n", epoch, i, finalOutput, targets[i], error);
            }
        }
        if (epoch % 100 == 0) {
            printf("Total Error at epoch %d: %f\n", epoch, totalError / (sizeof(targets)/sizeof(targets[0])));
        }
    }

    return 0;
}
