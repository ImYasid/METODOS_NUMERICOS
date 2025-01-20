# -*- coding: utf-8 -*-

import numpy as np

def gauss_seidel_iteraciones(A: np.ndarray, b: np.ndarray, x0: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales Ax = b mediante el método de Gauss-Seidel.

    ## Parameters

    ``A``: Matriz de coeficientes de dimensiones n x n.
    ``b``: Vector del término independiente de dimensión n.
    ``x0``: Vector inicial de dimensión n.
    ``tol``: Tolerancia para el criterio de parada.
    ``max_iter``: Número máximo de iteraciones.

    ## Return

    ``x``: Solución aproximada del sistema Ax = b.
    """
    n = len(b)
    x = x0.copy()

    for k in range(max_iter):
        x_new = x.copy()

        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        print("Iteración", k+1, ", solución aproximada:", x_new)

def gauss_seidel(A: np.ndarray, b: np.ndarray, x0: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales Ax = b mediante el método de Gauss-Seidel.

    ## Parameters

    ``A``: Matriz de coeficientes de dimensiones n x n.
    ``b``: Vector del término independiente de dimensión n.
    ``x0``: Vector inicial de dimensión n.
    ``tol``: Tolerancia para el criterio de parada.
    ``max_iter``: Número máximo de iteraciones.

    ## Return

    ``x``: Solución aproximada del sistema Ax = b.
    """
    n = len(b)
    x = x0.copy()

    for k in range(max_iter):
        x_new = x.copy()

        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        # Verificar el criterio de parada
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new

        x = x_new

    raise ValueError("El método de Gauss-Seidel no convergió.")

def es_diagonal_estrictamente_dominante(A: np.ndarray) -> bool:
    """Verifica si la matriz A es estrictamente diagonal dominante.

    ## Parameters

    ``A``: Matriz de coeficientes de dimensiones n x n.

    ## Return

    ``bool``: True si la matriz es estrictamente diagonal dominante, False en caso contrario.
    """
    n = A.shape[0]

    for i in range(n):
        suma_fila = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        if np.abs(A[i, i]) <= suma_fila:
            return False

    return True
