#include "../includes/dataReader.h"

void read_csv(const char* filePath, double*** features, double** target, int* numSamples) {
    FILE *file = fopen(filePath, "r");
    char line[MAX_LINE_LENGTH];
    int i = 0, j = 0;

    if (!file) {
        printf("No se pudo abrir el archivo %s.\n", filePath);
        return;
    }

    *numSamples = 0;
    while (fgets(line, MAX_LINE_LENGTH, file) != NULL) {
        (*numSamples)++;
    }
    rewind(file);

    // Asignación de memoria para features y target
    *features = (double **)malloc(*numSamples * sizeof(double *));
    *target = (double *)malloc(*numSamples * sizeof(double));
    for (i = 0; i < *numSamples; i++) {
        (*features)[i] = (double *)malloc(NUM_FEATURES * sizeof(double));
    }

    // Leer y almacenar los datos
    i = 0;
    while (fgets(line, MAX_LINE_LENGTH, file) != NULL && i < *numSamples) {
        char* token = strtok(line, ";");
        for (j = 0; j < NUM_FEATURES; j++) {
            (*features)[i][j] = atof(token);
            token = strtok(NULL, ";");
        }
        (*target)[i] = atof(token);
        i++;
    }

    //printf("%d", *numSamples);

    fclose(file);
}