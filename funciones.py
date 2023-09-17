"""
Created on Mon Sep 11 14:11:17 2023

@author: Bren Guzmán, María José Merino, Brenda García
"""

#Lib
import numpy as np

 #Distancia Manhattan
def distancia_manhattan(x1, y1, x2, y2, cov_matrix=None):
     #return abs(x1 - x2) + abs(y1 - y2)
     distancia = distancia_minkowski([x1], [x2], [y1], [y2], p=1)
     return distancia

 #Distancia de Euclidiana
def distancia_euclidiana(x1, y1, x2, y2, cov_matrix=None):
    distancia_euclidiana = distancia_minkowski([x1], [x2], [y1], [y2], p=2)
    #return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distancia_euclidiana
  
 #Distancia Mahalanobis
def distancia_mahalanobis(x1, x2, y1, y2, cov_matrix):
    # Verifica que cov_matrix tenga al menos dos dimensiones
    if cov_matrix.ndim < 2:
        raise ValueError("La matriz de covarianza debe tener al menos dos dimensiones")

    x_minus_y = np.array([x1, y1]) - np.array([x2, y2])
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mahalanobis_distance = np.sqrt(np.dot(np.dot(x_minus_y, inv_cov_matrix), x_minus_y.T))
    return mahalanobis_distance

 #Distancia Minkowski
def distancia_minkowski(x1, x2, y1, y2, p):
    x_minus_y = np.array([x1, y1]) - np.array([x2, y2])
    minkowski_distance = (np.sum(np.abs(x_minus_y) ** p)) ** (1/p)
    return minkowski_distance

 #Generar el diccionario con las distancias
def calcular_distancias(col_x, col_y, punto_referencia, dataframe, funcion_distancia, cov_matrix=None):
    # Columnas a utilizar para el cálculo de distancia
    x_column = col_x
    y_column = col_y
    
    # Inicializa un diccionario para almacenar las distancias
    distancias_dict = {}
    
    # Itera a través de los registros en el DataFrame
    for idx, row in dataframe.iterrows():
        x = row[x_column]
        y = row[y_column]
        
        # Calcula la distancia utilizando la función de distancia proporcionada
        distancia = funcion_distancia(punto_referencia[x_column], punto_referencia[y_column], x, y, cov_matrix)
 
        # Almacena la distancia en el diccionario
        distancias_dict[idx] = distancia

    return distancias_dict

 # K Vecinos Cercanos
def KNN(k, punto_referencia, diccionario, dataframe):
    # Eliminar elementos con distancia 0 (punto de referencia)
    diccionario = {clave: valor for clave, valor in diccionario.items() if valor > 0}
    
    # Ordenar el diccionario de distancias
    distancias_ordenadas = sorted(diccionario.items(), key=lambda x: x[1])
    
    # Seleccionar KNN
    k_vecinos = distancias_ordenadas[:k]
    
    # Buscar k vecinos cercanos en el df
    registros_vecinos = [dataframe.iloc[i] for i, _ in k_vecinos]
    
    # Imprimir los K vecinos más cercanos
    print(f"\n\nLos {k} vecinos más cercanos son:")
    for i, distancia in k_vecinos:
        print(f"Índice: {i}, Distancia: {distancia}, Clase: {dataframe.iloc[i]['class']}")
    
    # Retornar la clase estimada
    clases_vecinos = [vecino['class'] for vecino in registros_vecinos]
    clase_estimada = max(set(clases_vecinos), key=clases_vecinos.count)
    
    return clase_estimada
