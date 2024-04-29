#include "../includes/config.h"

int main() {
    unsigned int seed;
    double cpu_time_used;
    clock_t start, end;

    if (RtlGenRandom(&seed, sizeof(seed))) {
        srand(seed);
    }

    // Inicializar la MLP
    RNN *rnn = (RNN *)malloc(sizeof(RNN));
    rnn->lastError = 0;
    initializeRNN(rnn, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

    // Leer el dataset
    double **features = NULL;
    double *target = NULL;
    int numSamples = 0;
    //read_csv("../datasets/dataset_temp15_train.csv", &features, &target, &numSamples);

    FILE *file2 = fopen("../resultados-100.csv", "w");  // Abre un archivo en modo escritura
    if (file2 == NULL) {
        printf("Error al abrir el archivo.\n");
        return 1;  // Termina el programa si no se puede abrir el archivo
    }

    for (int j = 0; j < 10; ++j) {
        read_csv("../datasets/dataset_temp15_train.csv", &features, &target, &numSamples);
        rnn->lastError = 0;
        initializeRNN(rnn, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        start = clock();
        double totalError = 0.0;
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            totalError = 0.0;
            for (int i = 0; i < numSamples; i++) {
                totalError += trainMSE(rnn, features[i], target[i], LEARNING_RATE);
            }
            //if(epoch % 100 == 0)
            // Calcular y mostrar el error promedio y el promedio de error en porcentaje de la época
            //printf("Epoch %d - MSE: %f\n", epoch, totalError / numSamples);
        }
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC; // Convertir a segundos
        /*
        guardarPesosYBiasCSV(rnn, "../pesos_bias/pesos_capa_oculta.csv",
                             "../pesos_bias/bias_capa_oculta.csv",
                             "../pesos_bias/pesosState_capa_oculta.csv",
                             "../pesos_bias/pesos_capa_salida.csv",
                             "../pesos_bias/bias_capa_salida.csv",
                             "../pesos_bias/pesosState_capa_salida.csv");

        cargarPesosYBiasCSV(rnn, "../pesos_bias/pesos_capa_oculta.csv",
                             "../pesos_bias/bias_capa_oculta.csv",
                             "../pesos_bias/pesosState_capa_oculta.csv",
                             "../pesos_bias/pesos_capa_salida.csv",
                             "../pesos_bias/bias_capa_salida.csv",
                             "../pesos_bias/pesosState_capa_salida.csv");

        */
        read_csv("../datasets/dataset_temp15_test.csv", &features, &target, &numSamples);

        testRNN(rnn, features, target, numSamples, file2);

        fprintf(file2, ",%f\n", cpu_time_used);
        //printf("Tiempo de CPU utilizado: %f segundos\n", cpu_time_used);

    }

    //Liberar la memoria
    freeRNN(rnn);
    for (int i = 0; i < numSamples; i++) {
        free(features[i]);
    }
    free(features);
    free(target);

    return 0;
}
