# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 12:41:34 2023

@author: Bren Guzmán
"""
#%% LIBRERÍAS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from funciones import distancia_manhattan, distancia_euclidiana, distancia_mahalanobis
from funciones import calcular_distancias
from funciones import KNN
#%% CARGAR DATOS
nombres_columnas = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris = pd.read_csv("iris.data", names=nombres_columnas)

print(iris.head())

#%% SELECCIONAR 6 REGISTROS AL AZAR

# Dataframe para los puntos a clasificar
registros = pd.DataFrame(columns=iris.columns)

grupos = iris.groupby('class')

# Iteraciones
for nombre_clase, grupo in grupos:
    dos_registros = grupo.sample(n=2, random_state=3)
    registros = pd.concat([registros, dos_registros])

registros

#%% SELECCIONAR LAS CARACTERÍSTICAS A UTILIZAR

# Grid de gráficas de puntos para las cuatro características
sns.set(style="ticks")
sns.pairplot(iris, hue="class", markers=["o", "s", "D"])
plt.suptitle("Grid de Gráficas de Puntos por Clase", y=1.02)

plt.show()
# se utilizarán las características x= petal_length, y= petal_width

#%% KNN

#*** se pueden modificar desde aquí para intentar con distintas características ***
x = 'petal_length' 
y = 'petal_width'  

k = 3 #*** se puede modificar desde aquí para intentar con distintos k ***
data = iris[[x, y]]
cov_matrix= np.cov(data.values.T)

clases_estimadasE = []
clases_estimadasMn = []
clases_estimadasMs = []


for idx, punto_de_referencia in registros.iterrows():
    
    # EUCLIDIAN
    distancias_euclidiana = calcular_distancias(x, y, punto_de_referencia, iris, distancia_euclidiana)
    clase_e = KNN(k, punto_de_referencia, distancias_euclidiana, iris)    
    clases_estimadasE.append(clase_e)
    
    
    # MANHATTAN
    distancias_manhattan = calcular_distancias(x, y, punto_de_referencia, iris, distancia_manhattan)    
    clase_mn = KNN(k, punto_de_referencia, distancias_manhattan, iris)        
    clases_estimadasMn.append(clase_mn)
    
    
    # MAHALANOBIS
    distancias_mahalanobis = calcular_distancias(x, y, punto_de_referencia, iris, distancia_mahalanobis, cov_matrix)
    clase_ms = KNN(k, punto_de_referencia, distancias_mahalanobis, iris)    
    clases_estimadasMs.append(clase_ms)
    

registros['Euclidiana'] = clases_estimadasE
registros['Manhattan'] = clases_estimadasMn
registros['Mahalanobis'] = clases_estimadasMs

registros



#%% CSV
registros.to_csv('estimaciones.csv', index=True, index_label="Index")
#%% GRÁFICA
