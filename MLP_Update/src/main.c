#include "../includes/config.h"
#include "../includes/mlp.h"
#include "../includes/dataReader.h"

double calculateMAE(double *target, double *predicted, int numSamples) {
    double sum = 0.0;
    for (int i = 0; i < numSamples; i++) {
        sum += fabs(target[i] - predicted[i]);
    }
    return sum / numSamples;
}

int main() {
    srand((unsigned int)time(NULL));

    // Parámetros de la red y entrenamiento
    int inputSize = INPUT_SIZE;
    int hiddenSize = HIDDEN_SIZE;
    int outputSize = OUTPUT_SIZE;
    int epochs = 10000;
    double learningRate = LEARNING_RATE;

    // Inicializar la MLP
    MLP mlp;
    initializeMLP(&mlp, inputSize, hiddenSize, outputSize);

    // Leer el dataset
    double **features = NULL;
    double *target = NULL;
    int numSamples = 0;
    read_csv("../datasets/dataset_wine_normalizado.csv", &features, &target, &numSamples);
    initializeMLP(&mlp, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

    double totalErrorPercentage = 0.0;
    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalError = 0.0;
        totalErrorPercentage = 0.0; // Resetear para cada época
        for (int i = 0; i < numSamples; i++) {
            double hiddenOutputs[HIDDEN_SIZE];
            double predictedOutput = forward(&mlp, features[i], hiddenOutputs);
            train(&mlp, features[i], target[i], LEARNING_RATE);

            double error = fabs(target[i] - predictedOutput);
            totalError += error;

            if (0) {  // Mostrar solo para los primeros 5 ejemplos
                printf("Epoch %d, Sample %d - Target: %f, Predicted: %f, Error: %f\n", epoch, i, target[i], predictedOutput, error);
            }
        }
        // Calcular y mostrar el error promedio y el promedio de error en porcentaje de la época
        printf("Epoch %d - Average Total Error: %f\n", epoch, totalError / numSamples);
    }


    // No olvides liberar la memoria al final
    freeMLP(&mlp);
    for (int i = 0; i < numSamples; i++) {
        free(features[i]);
    }
    free(features);
    free(target);

    return 0;
}
