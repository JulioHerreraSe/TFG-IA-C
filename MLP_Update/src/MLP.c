#include "../includes/mlp.h"

double generateNormal() {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

double generateHeWeight(int previousLayerSize) {
    return generateNormal() * sqrt(2.0 / previousLayerSize);
}

double ReLU(double x){
    if(x > 0){
        return x;
    }else {
        return 0;
    }
}

double ReLUDerivative(double x) {
    if(x > 0){
        return 1;
    }else {
        return 0;
    }
}

static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

static double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

void initializeMLP(MLP *mlp, int inputSize, int hiddenSize, int outputSize) {
    int i, j;

    // Hidden Layer
    mlp->hiddenLayer.inputSize = inputSize;
    mlp->hiddenLayer.outputSize = hiddenSize;
    mlp->hiddenLayer.weights = (double **)malloc(hiddenSize * sizeof(double *));
    mlp->hiddenLayer.bias = (double *)malloc(hiddenSize * sizeof(double));
    for (i = 0; i < hiddenSize; i++) {
        mlp->hiddenLayer.weights[i] = (double *)malloc(inputSize * sizeof(double));
        for (j = 0; j < inputSize; j++) {
            mlp->hiddenLayer.weights[i][j] = generateHeWeight(inputSize);
        }
        mlp->hiddenLayer.bias[i] = generateNormal() * sqrt(2.0 / (inputSize + hiddenSize));
    }

    // Output Layer
    mlp->outputLayer.inputSize = hiddenSize;
    mlp->outputLayer.outputSize = outputSize;
    mlp->outputLayer.weights = (double **)malloc(outputSize * sizeof(double *));
    mlp->outputLayer.bias = (double *)malloc(outputSize * sizeof(double));
    for (i = 0; i < outputSize; i++) {
        mlp->outputLayer.weights[i] = (double *)malloc(hiddenSize * sizeof(double));
        for (j = 0; j < hiddenSize; j++) {
            mlp->outputLayer.weights[i][j] = generateHeWeight(hiddenSize);
        }
        mlp->outputLayer.bias[i] = generateNormal() * sqrt(2.0 / (hiddenSize + outputSize));
    }
}

double forward(MLP *mlp, double *input, double *hiddenOutput) {
    double finalOutput = 0.0;
    int i, j;

    // Calculate hidden layer output
    for (i = 0; i < mlp->hiddenLayer.outputSize; i++) {
        hiddenOutput[i] = 0.0;
        for (j = 0; j < mlp->hiddenLayer.inputSize; j++) {
            hiddenOutput[i] += input[j] * mlp->hiddenLayer.weights[i][j];
            //printf("input %d: %lf, peso: %lf ", j, input[j], mlp->hiddenLayer.weights[i][j]);
        }
        //printf("\n");
        hiddenOutput[i] += mlp->hiddenLayer.bias[i];
        hiddenOutput[i] = sigmoid(hiddenOutput[i]);
    }

    // Calculate final output
    for (i = 0; i < mlp->outputLayer.outputSize; i++) {
        finalOutput = 0.0;
        for (j = 0; j < mlp->outputLayer.inputSize; j++) {
            finalOutput += hiddenOutput[j] * mlp->outputLayer.weights[i][j];
        }
        finalOutput += mlp->outputLayer.bias[i];
        // In this case, we might not apply an activation function for regression
        // finalOutput = sigmoid(finalOutput); // Uncomment if needed
    }

    return finalOutput;
}

void train(MLP *mlp, double *input, double target, double learningRate) {
    int i, j;
    double hiddenOutputs[HIDDEN_SIZE];
    double predictedOutput = forward(mlp, input, hiddenOutputs);
    double *outputDeltas = (double *)malloc(OUTPUT_SIZE * sizeof(double));
    double *hiddenDeltas = (double *)malloc(HIDDEN_SIZE * sizeof(double));

    // Calcular delta para la capa de salida
    for (i = 0; i < OUTPUT_SIZE; i++) {
        outputDeltas[i] = (target - predictedOutput);  // Error derivativo para la función de pérdida MSE
    }

    // Actualizar pesos y bias de la capa de salida
    for (i = 0; i < OUTPUT_SIZE; i++) {
        for (j = 0; j < HIDDEN_SIZE; j++) {
            mlp->outputLayer.weights[i][j] += learningRate * outputDeltas[i] * hiddenOutputs[j];
        }
        mlp->outputLayer.bias[i] += learningRate * outputDeltas[i];
    }

    // Calcular delta para la capa oculta
    for (i = 0; i < HIDDEN_SIZE; i++) {
        hiddenDeltas[i] = 0;
        for (j = 0; j < OUTPUT_SIZE; j++) {
            hiddenDeltas[i] += outputDeltas[j] * mlp->outputLayer.weights[j][i];
        }
        hiddenDeltas[i] *= sigmoidDerivative(hiddenOutputs[i]);
    }

    // Actualizar pesos y bias de la capa oculta
    for (i = 0; i < HIDDEN_SIZE; i++) {
        for (j = 0; j < INPUT_SIZE; j++) {
            mlp->hiddenLayer.weights[i][j] += learningRate * hiddenDeltas[i] * input[j];
        }
        mlp->hiddenLayer.bias[i] += learningRate * hiddenDeltas[i];
    }

    // Liberar memoria de los deltas
    free(outputDeltas);
    free(hiddenDeltas);
}

void testMLP(const MLP *mlp, double **features, double *target, int numSamplesTest) {
    double sumPercentageError = 0.0;
    for (int i = 0; i < numSamplesTest; i++) {
        double hiddenOutputs[HIDDEN_SIZE]; // Asume que HIDDEN_SIZE es conocido aquí
        double predicted = forward(mlp, features[i], hiddenOutputs);
        double error = target[i] - predicted;

        // Asegúrate de que el target[i] no es cero para evitar la división por cero
        if (target[i] != 0) {
            sumPercentageError += fabs(error / target[i]);
        }
        printf("Test Sample %d - Target: %f, Predicted: %f, Error: %f\n", i, target[i], predicted, error);
    }
    double mape = (sumPercentageError / numSamplesTest) * 100; // Multiplica por 100 para obtener el porcentaje
    printf("MAPE (Test Set): %f%%\n", mape);
}


void guardarPesosYBiasCSV(const MLP *mlp, const char* archivoPesosCapaOculta, const char* archivoBiasCapaOculta, const char* archivoPesosCapaSalida, const char* archivoBiasCapaSalida) {
    int i, j;

    // Guardar pesos y bias de la capa oculta
    FILE* filePesos = fopen(archivoPesosCapaOculta, "w");
    FILE* fileBias = fopen(archivoBiasCapaOculta, "w");
    if (filePesos != NULL && fileBias != NULL) {
        for (i = 0; i < mlp->hiddenLayer.outputSize; i++) {
            for (j = 0; j < mlp->hiddenLayer.inputSize; j++) {
                fprintf(filePesos, "%f", mlp->hiddenLayer.weights[i][j]);
                if (j < mlp->hiddenLayer.inputSize - 1) fprintf(filePesos, ";");
            }
            fprintf(filePesos, "\n");
            fprintf(fileBias, "%f\n", mlp->hiddenLayer.bias[i]);
        }
        fclose(filePesos);
        fclose(fileBias);
    } else {
        printf("No se pudo abrir los archivos para escribir los pesos y bias de la capa oculta.\n");
    }

    // Guardar pesos y bias de la capa de salida
    filePesos = fopen(archivoPesosCapaSalida, "w");
    fileBias = fopen(archivoBiasCapaSalida, "w");
    if (filePesos != NULL && fileBias != NULL) {
        for (i = 0; i < mlp->outputLayer.outputSize; i++) {
            for (j = 0; j < mlp->outputLayer.inputSize; j++) {
                fprintf(filePesos, "%f", mlp->outputLayer.weights[i][j]);
                if (j < mlp->outputLayer.inputSize - 1) fprintf(filePesos, ";");
            }
            fprintf(filePesos, "\n");
            fprintf(fileBias, "%f\n", mlp->outputLayer.bias[i]);
        }
        fclose(filePesos);
        fclose(fileBias);
    } else {
        printf("No se pudo abrir los archivos para escribir los pesos y bias de la capa de salida.\n");
    }
}

void cargarPesosYBiasCSV(MLP *mlp, const char* archivoPesosCapaOculta, const char* archivoBiasCapaOculta, const char* archivoPesosCapaSalida, const char* archivoBiasCapaSalida) {
    FILE* file;
    int i, j;

    // Cargar pesos y bias de la capa oculta
    file = fopen(archivoPesosCapaOculta, "r");
    FILE* fileBias = fopen(archivoBiasCapaOculta, "r");
    if (file != NULL && fileBias != NULL) {
        for (i = 0; i < mlp->hiddenLayer.outputSize; i++) {
            for (j = 0; j < mlp->hiddenLayer.inputSize; j++) {
                fscanf(file, "%lf;", &mlp->hiddenLayer.weights[i][j]);
            }
            fscanf(fileBias, "%lf\n", &mlp->hiddenLayer.bias[i]);
        }
        fclose(file);
        fclose(fileBias);
    } else {
        printf("No se pudo abrir los archivos para leer los pesos y bias de la capa oculta.\n");
    }

    // Cargar pesos y bias de la capa de salida
    file = fopen(archivoPesosCapaSalida, "r");
    fileBias = fopen(archivoBiasCapaSalida, "r");
    if (file != NULL && fileBias != NULL) {
        for (i = 0; i < mlp->outputLayer.outputSize; i++) {
            for (j = 0; j < mlp->outputLayer.inputSize; j++) {
                fscanf(file, "%lf;", &mlp->outputLayer.weights[i][j]);
            }
            fscanf(fileBias, "%lf\n", &mlp->outputLayer.bias[i]);
        }
        fclose(file);
        fclose(fileBias);
    } else {
        printf("No se pudo abrir los archivos para leer los pesos y bias de la capa de salida.\n");
    }
}

void freeMLP(MLP *mlp) {
    int i;

    // Liberar capa oculta
    for (i = 0; i < mlp->hiddenLayer.outputSize; i++) {
        free(mlp->hiddenLayer.weights[i]);
    }
    free(mlp->hiddenLayer.weights);
    free(mlp->hiddenLayer.bias);

    // Liberar capa de salida
    for (i = 0; i < mlp->outputLayer.outputSize; i++) {
        free(mlp->outputLayer.weights[i]);
    }
    free(mlp->outputLayer.weights);
    free(mlp->outputLayer.bias);
}

