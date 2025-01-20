# -*- coding: utf-8 -*-

# ----------------------------- logging --------------------------
import logging
from sys import stdout
from datetime import datetime

import numpy as np

def multiplicar_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiplica dos matrices A y B si sus dimensiones son compatibles.

    Parameters
    ----------
    A : np.ndarray
        Primera matriz (m x n).
    B : np.ndarray
        Segunda matriz (n x p).

    Returns
    -------
    np.ndarray
        Matriz resultado de la multiplicación de A y B (m x p).

    Raises
    ------
    ValueError
        Si las dimensiones de las matrices no son compatibles para la multiplicación.
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("Las dimensiones de las matrices no son compatibles para la multiplicación. "
                         "El número de columnas de A debe ser igual al número de filas de B.")
    
    C = np.dot(A, B)
    
    return C

# ------------------------------------------------------------------------

def determinante(A: np.ndarray) -> float:
    """Calcula el determinante de una matriz cuadrada A.

    Parameters
    ----------
    A : np.ndarray
        Matriz cuadrada de dimensiones n x n.

    Returns
    -------
    float
        Determinante de la matriz A.

    Raises
    ------
    ValueError
        Si la matriz no es cuadrada.
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("A debe ser un array de numpy.")
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz debe ser cuadrada.")

    n = A.shape[0]
    A = A.astype(float)

    det = 1

    for i in range(n):
        max_row = np.argmax(np.abs(A[i:, i])) + i
        if A[max_row, i] == 0:
            return 0

        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            det *= -1

        det *= A[i, i]

        for j in range(i + 1, n):
            if A[j, i] != 0:
                factor = A[j, i] / A[i, i]
                A[j, i:] -= factor * A[i, i:]

    return det

#----------------------------------------------------------------------------------

def inversa(A: np.ndarray) -> np.ndarray:
    """Calcula la inversa de una matriz cuadrada A utilizando el método de Gauss-Jordan.

    Parameters
    ----------
    A : np.ndarray
        Matriz cuadrada de dimensiones n x n.

    Returns
    -------
    np.ndarray
        Matriz inversa de A.

    Raises
    ------
    ValueError
        Si la matriz no es cuadrada o si la matriz es singular (no tiene inversa).
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("A debe ser un array de numpy.")
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz debe ser cuadrada.")
    
    n = A.shape[0]
    A = A.astype(float)

    I = np.eye(n)
    
    AI = np.hstack((A, I))

    for i in range(n):
        max_row = np.argmax(np.abs(AI[i:, i])) + i
        if AI[max_row, i] == 0:
            raise ValueError("La matriz es singular y no tiene inversa.")
        
        if max_row != i:
            AI[[i, max_row]] = AI[[max_row, i]]
        
        pivot = AI[i, i]
        AI[i, :] = AI[i, :] / pivot
        
        for j in range(n):
            if j != i:
                factor = AI[j, i]
                AI[j, :] -= factor * AI[i, :]
    
    A_inv = AI[:, n:]
    
    return A_inv
