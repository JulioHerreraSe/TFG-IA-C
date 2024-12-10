#include "../includes/config.h"

int main() {
    clock_t start, end;
    double cpu_time_used;

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
    read_csv("../datasets/dataset_temp15_train.csv", &features, &target, &numSamples);

    FILE *file = fopen("../resultados-10-c-mlp.csv", "w");  // Abre un archivo en modo escritura
    if (file == NULL) {
        printf("Error al abrir el archivo.\n");
        return 1;  // Termina el programa si no se puede abrir el archivo
    }

    for (int j = 0; j < 10; ++j) {

        initializeMLP(&mlp, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        //read_csv("../datasets/dataset_temp15_train.csv", &features, &target, &numSamples);

        start = clock();
        double totalError = 0.0;
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            totalError = 0.0;
            for (int i = 0; i < numSamples; i++) {
                totalError += trainMSE(&mlp, features[i], target[i], LEARNING_RATE);
            }
        }
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC; // Convertir a segundos

        /*guardarPesosYBiasCSV(&mlp, "../pesos_bias/pesos_capa_oculta.csv", "../pesos_bias/bias_capa_oculta.csv",
                             "../pesos_bias/pesos_capa_salida.csv", "../pesos_bias/bias_capa_salida.csv");

        cargarPesosYBiasCSV(&mlp, "../pesos_bias/pesos_capa_oculta.csv", "../pesos_bias/bias_capa_oculta.csv",
                            "../pesos_bias/pesos_capa_salida.csv", "../pesos_bias/bias_capa_salida.csv");
        */
        //read_csv("../datasets/dataset_temp15_test.csv", &features, &target, &numSamples);

        testMLP(&mlp, features, target, numSamples, file);

        fprintf(file, ",%f\n", cpu_time_used);
        //printf("Tiempo de CPU utilizado: %f segundos\n", cpu_time_used);
    }
    //Liberar la memoria
    freeMLP(&mlp);
    for (int i = 0; i < numSamples; i++) {
        free(features[i]);
    }
    free(features);
    free(target);

    return 0;
}
