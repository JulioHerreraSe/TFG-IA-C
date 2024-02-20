#include "../includes/mlp.h"
#include "../includes/config.h"

static double randRange(double min, double max) {
    return (double)rand() / RAND_MAX * (max - min) + min;
}

static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

static double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

void initializeMLP(MLP *mlp, int inputSize, int hiddenSize, int outputSize) {
    int i, j;

    // Hidden Layer
    mlp->hiddenLayer.inputSize = inputSize;
    mlp->hiddenLayer.outputSize = hiddenSize;
    mlp->hiddenLayer.weights = (double **)malloc(hiddenSize * sizeof(double *));
    mlp->hiddenLayer.bias = (double *)malloc(hiddenSize * sizeof(double));
    for (i = 0; i < hiddenSize; i++) {
        mlp->hiddenLayer.weights[i] = (double *)malloc(inputSize * sizeof(double));
        for (j = 0; j < inputSize; j++) {
            mlp->hiddenLayer.weights[i][j] = randRange(-1.0, 1.0);
        }
        mlp->hiddenLayer.bias[i] = randRange(-1.0, 1.0);
    }

    // Output Layer
    mlp->outputLayer.inputSize = hiddenSize;
    mlp->outputLayer.outputSize = outputSize;
    mlp->outputLayer.weights = (double **)malloc(outputSize * sizeof(double *));
    mlp->outputLayer.bias = (double *)malloc(outputSize * sizeof(double));
    for (i = 0; i < outputSize; i++) {
        mlp->outputLayer.weights[i] = (double *)malloc(hiddenSize * sizeof(double));
        for (j = 0; j < hiddenSize; j++) {
            mlp->outputLayer.weights[i][j] = randRange(-1.0, 1.0);
        }
        mlp->outputLayer.bias[i] = randRange(-1.0, 1.0);
    }
}

double forward(MLP *mlp, double *input, double *hiddenOutput) {
    double finalOutput = 0.0;
    int i, j;

    // Calculate hidden layer output
    for (i = 0; i < mlp->hiddenLayer.outputSize; i++) {
        hiddenOutput[i] = 0.0;
        for (j = 0; j < mlp->hiddenLayer.inputSize; j++) {
            hiddenOutput[i] += input[j] * mlp->hiddenLayer.weights[i][j];
        }
        hiddenOutput[i] += mlp->hiddenLayer.bias[i];
        hiddenOutput[i] = sigmoid(hiddenOutput[i]);
    }

    // Calculate final output
    for (i = 0; i < mlp->outputLayer.outputSize; i++) {
        finalOutput = 0.0;
        for (j = 0; j < mlp->outputLayer.inputSize; j++) {
            finalOutput += hiddenOutput[j] * mlp->outputLayer.weights[i][j];
        }
        finalOutput += mlp->outputLayer.bias[i];
        // In this case, we might not apply an activation function for regression
        // finalOutput = sigmoid(finalOutput); // Uncomment if needed
    }

    return finalOutput;
}

void train(MLP *mlp, double *input, double target, double learningRate) {
    int i, j;
    double hiddenOutputs[HIDDEN_SIZE];
    double predictedOutput = forward(mlp, input, hiddenOutputs);
    double *outputDeltas = (double *)malloc(OUTPUT_SIZE * sizeof(double));
    double *hiddenDeltas = (double *)malloc(HIDDEN_SIZE * sizeof(double));

    // Calcular delta para la capa de salida
    for (i = 0; i < OUTPUT_SIZE; i++) {
        outputDeltas[i] = (target - predictedOutput);  // Error derivativo para la función de pérdida MSE
    }

    // Actualizar pesos y bias de la capa de salida
    for (i = 0; i < OUTPUT_SIZE; i++) {
        for (j = 0; j < HIDDEN_SIZE; j++) {
            mlp->outputLayer.weights[i][j] += learningRate * outputDeltas[i] * hiddenOutputs[j];
        }
        mlp->outputLayer.bias[i] += learningRate * outputDeltas[i];
    }

    // Calcular delta para la capa oculta
    for (i = 0; i < HIDDEN_SIZE; i++) {
        hiddenDeltas[i] = 0;
        for (j = 0; j < OUTPUT_SIZE; j++) {
            hiddenDeltas[i] += outputDeltas[j] * mlp->outputLayer.weights[j][i];
        }
        hiddenDeltas[i] *= sigmoidDerivative(hiddenOutputs[i]);
    }

    // Actualizar pesos y bias de la capa oculta
    for (i = 0; i < HIDDEN_SIZE; i++) {
        for (j = 0; j < INPUT_SIZE; j++) {
            mlp->hiddenLayer.weights[i][j] += learningRate * hiddenDeltas[i] * input[j];
        }
        mlp->hiddenLayer.bias[i] += learningRate * hiddenDeltas[i];
    }

    // Liberar memoria de los deltas
    free(outputDeltas);
    free(hiddenDeltas);
}

void freeMLP(MLP *mlp) {
    int i;

    // Liberar capa oculta
    for (i = 0; i < mlp->hiddenLayer.outputSize; i++) {
        free(mlp->hiddenLayer.weights[i]);
    }
    free(mlp->hiddenLayer.weights);
    free(mlp->hiddenLayer.bias);

    // Liberar capa de salida
    for (i = 0; i < mlp->outputLayer.outputSize; i++) {
        free(mlp->outputLayer.weights[i]);
    }
    free(mlp->outputLayer.weights);
    free(mlp->outputLayer.bias);
}

