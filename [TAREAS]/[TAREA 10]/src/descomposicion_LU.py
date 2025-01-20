# -*- coding: utf-8 -*-

# ----------------------------- logging --------------------------
import logging
from sys import stdout
from datetime import datetime
import numpy as np

# ####################################################################
def descomposicion_LU(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Realiza la descomposición LU de una matriz cuadrada A.
    [IMPORTANTE] No se realiza pivoteo.

    ## Parameters

    ``A``: matriz cuadrada de tamaño n-by-n.

    ## Return

    ``L``: matriz triangular inferior.

    ``U``: matriz triangular superior. Se obtiene de la matriz ``A`` después de aplicar la eliminación gaussiana.
    """

    A = np.array(A, dtype=float)  # convertir en float, porque si no, puede convertir como entero

    assert A.shape[0] == A.shape[1], "La matriz A debe ser cuadrada."
    n = A.shape[0]

    L = np.zeros((n, n), dtype=float)

    for i in range(0, n):  # loop por columna

        # --- deterimnar pivote
        if A[i, i] == 0:
            raise ValueError("No existe solución única.")

        # --- Eliminación: loop por fila
        L[i, i] = 1
        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - m * A[i, i:]
            L[j, i] = m

    if A[n - 1, n - 1] == 0:
        raise ValueError("No existe solución única.")

    return L, A


# ####################################################################
def resolver_LU(L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales mediante la descomposición LU.

    ## Parameters

    ``L``: matriz triangular inferior.

    ``U``: matriz triangular superior.

    ``b``: vector de términos independientes.

    ## Return

    ``solucion``: vector con la solución del sistema de ecuaciones lineales.
    """

    # Sustitución hacia adelante
    n = L.shape[0]
    y = np.zeros(n)
    y[0] = b[0] / L[0, 0]

    for i in range(1, n):
        suma = sum(L[i, j] * y[j] for j in range(i))
        y[i] = (b[i] - suma) / L[i, i]

    # Sustitución hacia atrás
    x = np.zeros(n, dtype=float)
    x[n - 1] = y[n - 1] / U[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        suma = sum(U[i, j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - suma) / U[i, i]

    return x
