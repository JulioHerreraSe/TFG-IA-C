// main.c
#include "../includes/rnn.h"
#include "../includes/config.h"

int main() {
    RNNLayer rnnLayer;
    int inputSize = INPUT_SIZE;
    int stateSize = HIDDEN_SIZE;
    int outputSize = OUTPUT_SIZE;

    // Inicializa la capa RNN
    initializeRNNLayer(&rnnLayer, inputSize, stateSize, outputSize);

    // Datos de entrenamiento sintéticos
    double input[INPUT_SIZE] = {0.5, -0.2, 0.1}; // Ejemplo de entrada
    double target = 0.4; // Valor objetivo

    // Entrenamiento simple para ilustración
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        double output[OUTPUT_SIZE];
        forwardRNNLayer(&rnnLayer, input, output);

        // Calcula el error (asumiendo una salida única para simplificar)
        double error = target - output[0];
        double outputError = 2 * error; // Gradiente del error para la salida

        RNNLayerGradients grads;
        // Inicialización y cálculo de gradientes omitidos por simplicidad...

        updateRNNLayerWeights(&rnnLayer, &grads, LEARNING_RATE);

        if (epoch % 1000 == 0) {
            printf("Epoch %d, Error: %f\n", epoch, error);
        }
    }

    freeRNNLayer(&rnnLayer);

    return 0;
}
