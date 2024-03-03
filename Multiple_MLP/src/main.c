#include "../includes/config.h"

int main() {
    // Inicializa la semilla aleatoria
    srand((unsigned)time(NULL));

    // Cargar el dataset
    const char* filePath = "../datasets/dataset_wine_train.csv";
    double **features;
    double *targets;
    int numSamples;
    readData(filePath, &features, &targets, &numSamples);

    // Inicializar la red neuronal
    NeuralNetwork nn;
    initializeNetwork(&nn);
    // Entrenamiento de la red
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double totalError = 0.0;
        for (int i = 0; i < numSamples; i++) {
            double predicted[OUTPUT_SIZE];
            forward(&nn, features[i], predicted);
            backpropagation(&nn, features[i], &targets[i], LEARNING_RATE);
            totalError += 0.5 * pow(targets[i] - predicted[0], 2); // Error cuadrático medio
        }
        if (epoch % 100 == 0) {
            printf("Epoch %d, Error: %f\n", epoch, totalError / numSamples);
        }
    }

    // Limpieza y liberación de recursos
    freeNetwork(&nn);
    for (int i = 0; i < numSamples; ++i) {
        free(features[i]);
    }
    free(features);
    free(targets);

    return 0;
}