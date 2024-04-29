#include "../includes/RNN.h"

double generateNormal() {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

double generateXavierWeight(int previousLayerSize, int currentLayerSize) {
    return generateNormal() * sqrt(1.0 / previousLayerSize);
}

double generateHeWeight(int previousLayerSize) {
    return generateNormal() * sqrt(2.0 / previousLayerSize);
}

static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

static double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

void initializeRNN(RNN *rnn, int inputSize, int hiddenSize, int outputSize) {
    int i, j;
    // Hidden Layer
    rnn->hiddenLayer.inputSize = inputSize;
    rnn->hiddenLayer.outputSize = hiddenSize;
    rnn->hiddenLayer.weights = (double **)malloc(hiddenSize * sizeof(double *));
    rnn->hiddenLayer.weightsState = (double *)malloc(hiddenSize * sizeof(double *));
    rnn->hiddenLayer.bias = (double *)malloc(hiddenSize * sizeof(double));
    rnn->hiddenLayer.state = (double *)malloc(hiddenSize * sizeof(double *));
    rnn->hiddenLayer.lastState = (double *)malloc(hiddenSize * sizeof(double *));

    /*for (i = 0; i < hiddenSize; ++i) {
        rnn->hiddenLayer.state[i] = 0;
        rnn->hiddenLayer.weightsState[i] = generateXavierWeight(inputSize, hiddenSize);
    }*/

    for (i = 0; i < hiddenSize; i++) {
        rnn->hiddenLayer.weights[i] = (double *)malloc(inputSize * sizeof(double));
        for (j = 0; j < inputSize; j++) {
            rnn->hiddenLayer.weights[i][j] = generateHeWeight(inputSize);
        }
        rnn->hiddenLayer.state[i] = 0;
        rnn->hiddenLayer.weightsState[i] = generateHeWeight(inputSize);
        rnn->hiddenLayer.bias[i] = generateNormal() * sqrt(2.0 / (inputSize + hiddenSize));
    }

    // Output Layer
    rnn->outputLayer.inputSize = hiddenSize;
    rnn->outputLayer.outputSize = outputSize;
    rnn->outputLayer.weights = (double **)malloc(outputSize * sizeof(double *));
    rnn->outputLayer.weightsState = (double *)malloc(outputSize * sizeof(double *));
    rnn->outputLayer.bias = (double *)malloc(outputSize * sizeof(double));
    rnn->outputLayer.state = (double *)malloc(outputSize * sizeof(double *));
    rnn->outputLayer.lastState = (double *)malloc(outputSize * sizeof(double *));

    rnn->outputLayer.state[OUTPUT_SIZE-1] = 0;

    for (i = 0; i < outputSize; i++) {
        rnn->outputLayer.weights[i] = (double *)malloc(hiddenSize * sizeof(double));
        for (j = 0; j < hiddenSize; j++) {
            rnn->outputLayer.weights[i][j] = generateHeWeight(hiddenSize);
        }
        rnn->outputLayer.weightsState[i] = generateHeWeight(hiddenSize);
        rnn->outputLayer.bias[i] = generateNormal() * sqrt(2.0 / (hiddenSize + outputSize));
    }
}

// La función forward modificada para manejar secuencias de tiempo.
// Agrega un parámetro step para indicar el paso de tiempo actual.
double forward(RNN *rnn, double *input, double hiddenOutput[HIDDEN_SIZE]) {
    int i, j;
    // Calcular la salida de la capa oculta para el paso de tiempo actual.
    for (i = 0; i < rnn->hiddenLayer.outputSize; i++) {
        hiddenOutput[i] = rnn->hiddenLayer.bias[i];
        for (j = 0; j < rnn->outputLayer.inputSize; j++) {
            hiddenOutput[i] += input[j] * rnn->hiddenLayer.weights[i][j];
        }

            hiddenOutput[i] += rnn->hiddenLayer.state[i] * rnn->hiddenLayer.weightsState[i];

        hiddenOutput[i] = sigmoid(hiddenOutput[i]);
    }

    // Almacenar el estado oculto actual para su uso en el próximo paso de tiempo.
    for (i = 0; i < rnn->hiddenLayer.outputSize; ++i) {
        rnn->hiddenLayer.lastState[i] = rnn->hiddenLayer.state[i];
        rnn->hiddenLayer.state[i] = hiddenOutput[i];
    }

    // Calcular la salida de la red.
    double finalOutput = rnn->outputLayer.bias[OUTPUT_SIZE-1];
    for (i = 0; i < rnn->hiddenLayer.outputSize; i++) {
        finalOutput += hiddenOutput[i] * rnn->outputLayer.weights[OUTPUT_SIZE-1][i];
    }

    finalOutput += rnn->outputLayer.state[OUTPUT_SIZE-1] * rnn->outputLayer.weightsState[OUTPUT_SIZE-1];

    // Guardar el output actual como el estado de la capa de salida para este paso.
    rnn->outputLayer.lastState[OUTPUT_SIZE-1] = rnn->outputLayer.state[OUTPUT_SIZE-1];
    rnn->outputLayer.state[OUTPUT_SIZE-1] = finalOutput;

    return finalOutput;
}


double trainMSE(RNN *rnn, double *input, double target, double learningRate) {
    int i, j;
    double hiddenOutputs[HIDDEN_SIZE]; // Asegúrate de que HIDDEN_SIZE esté definido correctamente.
    double predictedOutput = forward(rnn, input, hiddenOutputs);
    double error = target - predictedOutput; // El error es el mismo para el cálculo
    double mse_gradient = error * 2; // Gradiente para MSE
    double mse_lastGradient = rnn->lastError * 2; // Gradiente para MSE

    // Calcula el delta para la capa de salida usando MSE
    double outputDelta = mse_gradient;
    double lastOutputDelta = mse_lastGradient;

    // Actualiza los pesos y bias de la capa de salida
    for (i = 0; i < OUTPUT_SIZE; i++) {
        for (j = 0; j < HIDDEN_SIZE; j++) {
            rnn->outputLayer.weights[i][j] += learningRate * outputDelta * hiddenOutputs[j];
        }
        rnn->outputLayer.bias[i] += learningRate * outputDelta;
    }

    rnn->outputLayer.weightsState[OUTPUT_SIZE-1] = learningRate * lastOutputDelta * rnn->outputLayer.state[OUTPUT_SIZE-1];


    // Calcula los deltas para la capa oculta
    double hiddenDeltas[HIDDEN_SIZE];
    double lastHiddenDeltas[HIDDEN_SIZE];
    for (i = 0; i < HIDDEN_SIZE; i++) {
        hiddenDeltas[i] = 0;
        lastHiddenDeltas[i] = 0;
        for (j = 0; j < OUTPUT_SIZE; j++) {
            hiddenDeltas[i] += outputDelta * rnn->outputLayer.weights[j][i];
            lastHiddenDeltas[i] += lastOutputDelta * rnn->outputLayer.weightsState[j];
        }
        // El uso de sigmoidDerivative sigue siendo válido aquí porque la propagación hacia atrás del error no cambia con la función de pérdida
        hiddenDeltas[i] *= sigmoidDerivative(hiddenOutputs[i]);

        lastHiddenDeltas[i] *= sigmoidDerivative(rnn->hiddenLayer.state[i]);

    }

    // Actualiza los pesos y bias de la capa oculta
    for (i = 0; i < HIDDEN_SIZE; i++) {
        for (j = 0; j < INPUT_SIZE; j++) {
            rnn->hiddenLayer.weights[i][j] += learningRate * hiddenDeltas[i] * input[j];
        }

        rnn->hiddenLayer.bias[i] += learningRate * hiddenDeltas[i];

        rnn->hiddenLayer.weightsState[i] += learningRate * lastHiddenDeltas[i] * rnn->hiddenLayer.lastState[i];

    }
    rnn->lastError = error;

    // Devuelve el error cuadrático para esta iteración de entrenamiento
    return error * error; // Devuelve MSE, que es el error al cuadrado

}


void testRNN(const RNN *rnn, double **features, double *target, int numSamplesTest, FILE *file) {
    double sumPercentageError = 0.0;
    double sumAbsoluteError = 0.0;
    double sumSquaredError = 0.0;
    double sumTarget = 0.0;
    double sumSquaredTarget = 0.0;
    double sumTargetPredicted = 0.0;
    for (int i = 0; i < numSamplesTest; i++) {

        double hiddenOutputs[HIDDEN_SIZE]; // Asume que HIDDEN_SIZE es conocido aquí
        double predicted = forward(rnn, features[i], hiddenOutputs);
        double error = target[i] - predicted;

        // Calcula MAE y MSE
        sumAbsoluteError += fabs(error);
        sumSquaredError += error * error;

        // Acumulación para R^2
        sumTarget += target[i];
        sumSquaredTarget += target[i] * target[i];
        sumTargetPredicted += target[i] * predicted;


        // MAPE
        if (target[i] != 0) {
            sumPercentageError += fabs(error / target[i]);
        }
        //printf("Test Sample %d - Target: %f, Predicted: %f, Error: %f\n", i, target[i], predicted, error);
    }
    double mae = sumAbsoluteError / numSamplesTest;
    double mse = sumSquaredError / numSamplesTest;

    double meanTarget = sumTarget / numSamplesTest;
    double ssTotal = sumSquaredTarget - numSamplesTest * meanTarget * meanTarget;
    double ssReg = sumTargetPredicted - numSamplesTest * meanTarget * meanTarget;
    double rSquared = ssReg / ssTotal;
    double mape = ((sumPercentageError / numSamplesTest) * 100);

    fprintf(file, "%f,%f,%f,%f",mape, mae, mse, rSquared);
    //printf("MAPE (Test Set): %f%%\n", (sumPercentageError / numSamplesTest) * 100);
    //printf("MAE (Test Set): %f\n", mae);
    //printf("MSE (Test Set): %f\n", mse);
    //printf("R-squared (Test Set): %f\n", rSquared);
}


void guardarPesosYBiasCSV(const RNN *rnn, const char* archivoPesosCapaOculta, const char* archivoBiasCapaOculta,
                          const char* archivoPesosStateCapaOculta, const char* archivoPesosCapaSalida,
                          const char* archivoBiasCapaSalida, const char* archivoPesosStateCapaSalida) {
    int i, j;

    // Guardar pesos y bias de la capa oculta
    FILE* filePesos = fopen(archivoPesosCapaOculta, "w");
    FILE* filePesosState = fopen(archivoPesosStateCapaOculta, "w");
    FILE* fileBias = fopen(archivoBiasCapaOculta, "w");
    if (filePesos != NULL && fileBias != NULL) {
        for (i = 0; i < rnn->hiddenLayer.outputSize; i++) {
            for (j = 0; j < rnn->hiddenLayer.inputSize; j++) {
                fprintf(filePesos, "%f", rnn->hiddenLayer.weights[i][j]);
                if (j < rnn->hiddenLayer.inputSize - 1) fprintf(filePesos, ";");
            }
            fprintf(filePesos, "\n");
            fprintf(filePesosState, "%f", rnn->hiddenLayer.weightsState[i]);
            if (i < rnn->hiddenLayer.outputSize - 1) fprintf(filePesosState, ";");
            fprintf(fileBias, "%f\n", rnn->hiddenLayer.bias[i]);
        }
        fprintf(filePesosState, "\n");
        fclose(filePesos);
        fclose(filePesosState);
        fclose(fileBias);
    } else {
        printf("No se pudo abrir los archivos para escribir los pesos y bias de la capa oculta.\n");
    }

    // Guardar pesos y bias de la capa de salida
    filePesos = fopen(archivoPesosCapaSalida, "w");
    filePesosState = fopen(archivoPesosStateCapaSalida, "w");
    fileBias = fopen(archivoBiasCapaSalida, "w");
    if (filePesos != NULL && fileBias != NULL) {
        for (i = 0; i < rnn->outputLayer.outputSize; i++) {
            for (j = 0; j < rnn->outputLayer.inputSize; j++) {
                fprintf(filePesos, "%f", rnn->outputLayer.weights[i][j]);
                if (j < rnn->outputLayer.inputSize - 1) fprintf(filePesos, ";");
            }
            fprintf(filePesos, "\n");
            fprintf(filePesosState, "%f\n", rnn->outputLayer.weightsState[i]);
            fprintf(fileBias, "%f\n", rnn->outputLayer.bias[i]);
        }

        fclose(filePesos);
        fclose(filePesosState);
        fclose(fileBias);
    } else {
        printf("No se pudo abrir los archivos para escribir los pesos y bias de la capa de salida.\n");
    }
}
void cargarPesosYBiasCSV(RNN *rnn, const char* archivoPesosCapaOculta, const char* archivoBiasCapaOculta,
                         const char* archivoPesosStateCapaOculta, const char* archivoPesosCapaSalida,
                         const char* archivoBiasCapaSalida, const char* archivoPesosStateCapaSalida) {
    FILE* filePesos, *fileBias, *filePesosState;
    int i, j;

    // Cargar pesos y bias de la capa oculta
    filePesos = fopen(archivoPesosCapaOculta, "r");
    fileBias = fopen(archivoBiasCapaOculta, "r");
    filePesosState = fopen(archivoPesosStateCapaOculta, "r");
    if (filePesos != NULL && fileBias != NULL && filePesosState != NULL) {
        for (i = 0; i < rnn->hiddenLayer.outputSize; i++) {
            for (j = 0; j < rnn->hiddenLayer.inputSize; j++) {
                fscanf(filePesos, "%lf;", &rnn->hiddenLayer.weights[i][j]);
            }
            fscanf(filePesosState, "%lf;", &rnn->hiddenLayer.weightsState[i]);
            fscanf(fileBias, "%lf\n", &rnn->hiddenLayer.bias[i]);
        }
        fclose(filePesos);
        fclose(filePesosState);
        fclose(fileBias);
    } else {
        printf("No se pudo abrir los archivos para leer los pesos y bias de la capa oculta.\n");
    }

    // Cargar pesos y bias de la capa de salida
    filePesos = fopen(archivoPesosCapaSalida, "r");
    fileBias = fopen(archivoBiasCapaSalida, "r");
    filePesosState = fopen(archivoPesosStateCapaSalida, "r");
    if (filePesos != NULL && fileBias != NULL && filePesosState != NULL) {
        for (i = 0; i < rnn->outputLayer.outputSize; i++) {
            for (j = 0; j < rnn->outputLayer.inputSize; j++) {
                fscanf(filePesos, "%lf;", &rnn->outputLayer.weights[i][j]);
            }
            fscanf(filePesosState, "%lf\n", &rnn->outputLayer.weightsState[i]);
            fscanf(fileBias, "%lf\n", &rnn->outputLayer.bias[i]);
        }
        fclose(filePesos);
        fclose(filePesosState);
        fclose(fileBias);
    } else {
        printf("No se pudo abrir los archivos para leer los pesos y bias de la capa de salida.\n");
    }
}


void freeRNN(RNN *rnn) {
    int i;

    // Liberar capa oculta
    for (i = 0; i < rnn->hiddenLayer.outputSize; i++) {
        free(rnn->hiddenLayer.weights[i]);
    }
    free(rnn->hiddenLayer.weights);
    free(rnn->hiddenLayer.weightsState);
    free(rnn->hiddenLayer.bias);

    // Liberar capa de salida
    for (i = 0; i < rnn->outputLayer.outputSize; i++) {
        free(rnn->outputLayer.weights[i]);
    }
    free(rnn->outputLayer.weights);
    free(rnn->outputLayer.weightsState);
    free(rnn->outputLayer.bias);
    free(rnn);
}

