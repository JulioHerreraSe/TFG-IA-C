#include <stdio.h>

// Función para encontrar el MCD y los coeficientes de Bézout (x, y) tal que ax + by = mcd(a, b)
void euclidesExtendido(int a, int b, int *x, int *y, int *mcd) {
    // Caso base
    if (b == 0) {
        *mcd = a;
        *x = 1;
        *y = 0;
        return;
    }

    int x1, y1; // Para almacenar resultados de llamadas recursivas
    euclidesExtendido(b, a % b, &x1, &y1, mcd);

    // Actualizar x y y usando los resultados de la llamada recursiva
    *x = y1;
    *y = x1 - (a / b) * y1;
}

int main() {
    int a, b, x, y, mcd;
    // Ejemplo de entrada
    printf("Ingrese los valores de a y b: ");
    scanf("%d %d", &a, &b);

    euclidesExtendido(a, b, &x, &y, &mcd);

    printf("El MCD de %d y %d es %d\n", a, b, mcd);
    printf("Los coeficientes de Bézout son x = %d, y = %d\n", x, y);

    return 0;
}
