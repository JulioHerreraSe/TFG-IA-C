#include "../includes/utils.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

double generateNormal() {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

double generateHeWeight(int previousLayerSize) {
    return generateNormal() * sqrt(2.0 / previousLayerSize);
}

void readData(const char* filePath, double*** features, double** target, int* numSamples) {
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
        (*features)[i] = (double *)malloc(INPUT_SIZE  * sizeof(double));
    }

    // Leer y almacenar los datos
    i = 0;
    while (fgets(line, MAX_LINE_LENGTH, file) != NULL && i < *numSamples) {
        char* token = strtok(line, ";");
        for (j = 0; j < INPUT_SIZE ; j++) {
            (*features)[i][j] = atof(token);
            token = strtok(NULL, ";");
        }
        (*target)[i] = atof(token);
        i++;
    }

    fclose(file);
}