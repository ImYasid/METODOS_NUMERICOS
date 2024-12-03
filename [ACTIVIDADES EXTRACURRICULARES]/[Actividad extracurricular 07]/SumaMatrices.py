def sumar_matrices(matriz1, matriz2):
    """
    Suma dos matrices del mismo tamaño.

    Parámetros:
    matriz1: Lista de listas, representa la primera matriz.
    matriz2: Lista de listas, representa la segunda matriz.

    Retorna:
    Una nueva matriz resultante de la suma de matriz1 y matriz2.
    """
    if len(matriz1) != len(matriz2) or any(
            len(fila1) != len(fila2)
            for fila1, fila2 in zip(matriz1, matriz2)):
        raise ValueError("Las matrices deben tener las mismas dimensiones")

    # Sumar elemento a elemento
    matriz_resultante = [[
        matriz1[i][j] + matriz2[i][j] for j in range(len(matriz1[0]))
    ] for i in range(len(matriz1))]
    return matriz_resultante


# Ejemplo de uso
matriz_a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

matriz_b = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]

resultado = sumar_matrices(matriz_a, matriz_b)

print("Resultado de la suma:")
for fila in resultado:
    print(fila)
