#include "../includes/config.h"

int main() {

    unsigned int seed;
    if (RtlGenRandom(&seed, sizeof(seed))) {
        srand(seed);
    }

    // Inicializar la MLP
    MLP mlp;
    initializeMLP(&mlp, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

    // Leer el dataset
    double **features = NULL;
    double *target = NULL;
    int numSamples = 0;
    //read_csv("../datasets/dataset_wine_train_target.csv", &features, &target, &numSamples);
    initializeMLP(&mlp, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    /*
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double totalError = 0.0;
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
        if(epoch % 100 == 0)
        // Calcular y mostrar el error promedio y el promedio de error en porcentaje de la época
        printf("Epoch %d - Average Total Error: %f\n", epoch, totalError / numSamples);
    }

    guardarPesosYBiasCSV(&mlp, "../pesos_bias/pesos_capa_oculta.csv", "../pesos_bias/bias_capa_oculta.csv", "../pesos_bias/pesos_capa_salida.csv", "../pesos_bias/bias_capa_salida.csv");
    */
    cargarPesosYBiasCSV(&mlp, "../pesos_bias/pesos_capa_oculta.csv", "../pesos_bias/bias_capa_oculta.csv", "../pesos_bias/pesos_capa_salida.csv", "../pesos_bias/bias_capa_salida.csv");

    read_csv("../datasets/dataset_wine_test_target.csv", &features, &target, &numSamples);

    testMLP(&mlp, features, target, numSamples);

    //Liberar la memoria
    freeMLP(&mlp);
    for (int i = 0; i < numSamples; i++) {
        free(features[i]);
    }
    free(features);
    free(target);

    return 0;
}
