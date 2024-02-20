#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void inicializarGeneradorAleatorio() {
    srand((unsigned)time(NULL));
}

// Genera un número aleatorio con distribución normal estándar
double generarNormal() {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// Genera un peso aleatorio usando la inicialización de He
double generarPesoHe(int tamanoCapaAnterior) {
    return generarNormal() * sqrt(2.0 / tamanoCapaAnterior);
}

double ReLU (double numero){
    if(numero > 0){
        return numero;
    }else {
        return 0;
    }
}

void multiplyMatrices(double A[][4], double B[][4], double C[][4], int A_rows, int A_cols, int B_rows, int B_cols) {
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            C[i][j] = 0; // Inicializar el elemento de la matriz resultante
            for (int k = 0; k < A_cols; k++) { // A_cols y B_rows son iguales
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {

    inicializarGeneradorAleatorio();

    double E = M_E;

    double entradas[3][4] = {{1.0,2.0,3.0, 2.5},
                            {2.0, 5.0, -1.0,2.0},
                            {-1.5, 2.7, 3.3, -0.8}};

    double pesos[3][4] = {{0.2, 0.8, -0.5, 1.0},
                       {0.5, -0.91, 0.26, -0.5},
                       {-0.26, -0.27, 0.17, 0.87}};
    double pesos2[3][3] = {{0.1, -0.14, 0.5},
                        {-0.5, 0.12, -0.33},
                        {-0.44,0.73,-0.13}};

    double pesosTraspuesta[4][3];
    double pesos2Traspuesta[3][3];
    double bias[] = {2.0,3.0,0.5};
    double bias2[] = {-1.0,2.0,-0.5};
    double salidas[3][3];
    double salidasTraspuesta[3][3];
    double salidas2[3][3];
    double salidas2Traspuesta[3][3];
    double salidasFinales[1][3];

    //Traspuesta
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 4; j++){
            pesosTraspuesta[j][i] = pesos[i][j];
        }
    }

    //Multiplicación de matrices
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            salidas[i][j] = bias[i];
            for (int k = 0; k < 4; k++) {
                salidas[i][j] += (entradas[i][k] * pesosTraspuesta[k][j]);
            }
        }
    }

    //Traspuesta
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            salidasTraspuesta[j][i] = salidas[i][j];
        }
    }

    printf("Matriz resultante:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%lf ", salidasTraspuesta[i][j]);
        }
        printf("\n");
    }

    multiplyMatrices(entradas, pesosTraspuesta, salidas, 3, 4, 4, 3);

    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            salidasTraspuesta[j][i] = salidas[i][j];
        }
    }

    printf("Matriz resultante C de A x B:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%f ", salidasTraspuesta[i][j]);
        }
        printf("\n");
    }

    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            pesos2Traspuesta[j][i] = pesos2[i][j];
        }
    }

    //Multiplicación de matrices
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            salidas2[i][j] = bias2[i];
            for (int k = 0; k < 3; k++) {
                salidas2[i][j] += salidasTraspuesta[i][k] * pesos2Traspuesta[k][j];
            }
        }
    }

    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            salidas2Traspuesta[j][i] = salidas2[i][j];
        }
    }

    printf("\nMatriz resultante:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%lf ", salidas2Traspuesta[i][j]);
        }
        printf("\n");
    }

    multiplyMatrices(salidas2, pesos2Traspuesta, salidasTraspuesta, 3, 3, 3, 3);

    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            salidas2Traspuesta[j][i] = salidas2[i][j];
        }
    }

    printf("Matriz resultante C de A x B:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%f ", salidas2Traspuesta[i][j]);
        }
        printf("\n");
    }

    inicializarGeneradorAleatorio();

    const int numPesos = 10;
    int tamanoCapaAnterior = 4; // Ejemplo: 4 neuronas en la capa anterior

    printf("Pesos aleatorios generados con inicialización de He:\n");
    for (int i = 0; i < numPesos; i++) {
        double peso = generarPesoHe(tamanoCapaAnterior);
        printf("%f\n", peso);
    }

    return 0;
}
