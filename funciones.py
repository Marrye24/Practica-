"""
Created on Mon Sep 11 14:11:17 2023

@author: Bren Guzmán, María José Merino, Brenda García
"""

#Distancia Manhattan y Minkowski

import math

def distancia_minkowski(a, b, p):

    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma dimensión")

    if p <= 0:
        raise ValueError("El parámetro p debe ser mayor que 0")

    distancia = 0
    for i in range(len(a)):
        distancia += abs(a[i] - b[i]) ** p

    distancia = distancia ** (1/p)
    return distancia

# Ejemplo de uso
vector_a = [1.7,0.3]
vector_b = [2.5,0.5]
p = 4  # Esto corresponde a la distancia euclidiana

dist_euclidiana = distancia_minkowski(vector_a, vector_b, p)
print(f"La distancia euclidiana entre los vectores es: {dist_euclidiana}")

def distancia_manhattan(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

# Ejemplo de uso:
punto1 = (1.7,0.3)
punto2 = (2.5,0.5)

distancia = distancia_manhattan(punto1[0], punto1[1], punto2[0], punto2[1])
print(f"La distancia de Manhattan entre {punto1} y {punto2} es {distancia}")

#Distancia Mahalanobis y Euclidiana

import numpy as np
import math

def calcular_covarianza(vectores):
    """
    Calcula la matriz de covarianza entre un conjunto de vectores.

    :param vectores: Una lista de vectores, donde cada vector es una lista de características.
    :return: La matriz de covarianza.
    """
    if len(vectores) == 0:
        raise ValueError("La lista de vectores no puede estar vacía")

    n = len(vectores[0])  # Dimensión de los vectores
    m = len(vectores)     # Número de vectores

    # Calcula la media de cada característica
    medias = [sum(v[i] for v in vectores) / m for i in range(n)]

    # Calcula la matriz de covarianza
    covarianza = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov_ij = sum((vectores[k][i] - medias[i]) * (vectores[k][j] - medias[j]) for k in range(m))
            covarianza[i][j] = cov_ij / (m - 1)

    return covarianza

def distancia_mahalanobis(x, y, cov_matrix):
    
    if len(x) != len(y) or len(x) != len(cov_matrix) or len(cov_matrix) != len(cov_matrix[0]):
        raise ValueError("Las dimensiones de los vectores y la matriz de covarianza deben coincidir")

    x_minus_y = np.array(x) - np.array(y)
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    mahalanobis_distance = np.sqrt(np.dot(np.dot(x_minus_y, inv_cov_matrix), x_minus_y.T))
    return mahalanobis_distance

# Ejemplo de uso
vector1 = [1.7,0.3]
vector2 = [1.2,3.0]

# Crear una lista de vectores de datos
vectores = [vector1, vector2]

# Calcular la matriz de covarianza
cov_matrix = calcular_covarianza(vectores)

# Calcular la distancia Mahalanobis entre vector1 y vector2 usando la matriz de covarianza
dist_mahalanobis = distancia_mahalanobis(vector1, vector2, cov_matrix)

print(f"La distancia Mahalanobis entre vector1 y vector2 es: {dist_mahalanobis}")

def distancia_euclidiana(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Ejemplo de uso:
punto1 = (1.7,0.3)
punto2 = (2.5,0.5)

distancia = distancia_euclidiana(punto1[0], punto1[1], punto2[0], punto2[1])
print(f"La distancia euclidiana entre {punto1} y {punto2} es {distancia}")

