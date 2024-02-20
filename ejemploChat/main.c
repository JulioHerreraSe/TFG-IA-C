#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 3 // Número de características de entrada
#define HIDDEN_SIZE 4 // Número de neuronas en la capa oculta
#define OUTPUT_SIZE 1 // Una sola salida para regresión

typedef struct {
    double weights[INPUT_SIZE][HIDDEN_SIZE]; // Pesos de entrada a la capa oculta
    double bias[HIDDEN_SIZE]; // Bias de la capa oculta
} HiddenLayer;

typedef struct {
    double weights[HIDDEN_SIZE][OUTPUT_SIZE]; // Pesos de la capa oculta a la salida
    double bias[OUTPUT_SIZE]; // Bias de la capa de salida
} OutputLayer;

typedef struct {
    HiddenLayer hiddenLayer;
    OutputLayer outputLayer;
} MLP;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double mse(double output[], double target[], int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        double error = output[i] - target[i];
        sum += error * error;
    }
    return sum / size;
}
void forward(MLP *mlp, double input[], double output[]) {
    double hiddenActivations[HIDDEN_SIZE];

    // Capa oculta
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hiddenActivations[i] = mlp->hiddenLayer.bias[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            hiddenActivations[i] += input[j] * mlp->hiddenLayer.weights[j][i];
        }
        hiddenActivations[i] = sigmoid(hiddenActivations[i]); // Usando sigmoid como ejemplo
    }

    // Capa de salida
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = mlp->outputLayer.bias[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output[i] += hiddenActivations[j] * mlp->outputLayer.weights[j][i];
        }
        // No aplicamos función de activación en la salida para regresión
    }
}
int main() {
    MLP mlp; // Debería inicializarse adecuadamente

    double input[INPUT_SIZE] = {1.0, 0.5, -1.2}; // Ejemplo de entrada
    double output[OUTPUT_SIZE]; // Almacena la salida de la red

    forward(&mlp, input, output); // Realiza la propagación hacia adelante

    double target[OUTPUT_SIZE] = {0.3}; // Valor objetivo para calcular el error
    printf("MSE: %f\n", mse(output, target, OUTPUT_SIZE));

    return 0;
}
