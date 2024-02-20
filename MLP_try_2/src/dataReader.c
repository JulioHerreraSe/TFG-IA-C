#include "../includes/dataReader.h"

void read_csv() {
    FILE *file;
    char line[MAX_LINE_LENGTH];
    double **data; // Matriz para las características
    double *target; // Vector para la columna objetivo
    int numSamples = 0;
    int i, j;

    // Abrir el archivo
    file = fopen("../datasets/dataset_wine_normalizado.csv", "r");
    if (!file) {
        printf("No se pudo abrir el archivo.\n");
    }

    // Contar el número de líneas para determinar numSamples
    while (fgets(line, MAX_LINE_LENGTH, file) != NULL) {
        numSamples++;
    }
    rewind(file); // Regresar al inicio del archivo

    // Asignar memoria para data y target
    data = (double **)malloc(numSamples * sizeof(double *));
    for (i = 0; i < numSamples; i++) {
        data[i] = (double *)malloc(NUM_FEATURES * sizeof(double));
    }
    target = (double *)malloc(numSamples * sizeof(double));

    // Leer y procesar cada línea
    i = 0;
    while (fgets(line, MAX_LINE_LENGTH, file) != NULL) {
        // Separar los valores por comas
        char *token = strtok(line, ";");
        for (j = 0; j < NUM_FEATURES; j++) {
            if (token != NULL) {
                data[i][j] = atof(token);
                token = strtok(NULL, ";");
            }
        }
        // La última columna es el target
        if (token != NULL) {
            target[i] = atof(token);
        }
        i++;
    }

    fclose(file);

    // Opcional: Imprimir algunos datos para verificar
    for (i = 0; i < 5; i++) { // Imprimir solo las primeras 5 muestras
        for (j = 0; j < NUM_FEATURES; j++) {
            printf("%f ", data[i][j]);
        }
        printf("| %f\n", target[i]);
    }

    // Liberar memoria
    for (i = 0; i < numSamples; i++) {
        free(data[i]);
    }
    free(data);
    free(target);
}
