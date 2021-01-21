#! /usr/bin/env python3
# -*- coding: utf-8 -*-
################################################################################
# Librerías
################################################################################

import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocesamiento
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Algoritmos
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# Evaluación
from sklearn.metrics import classification_report, \
                            confusion_matrix, \
                            plot_confusion_matrix

################################################################################
# Lectura
################################################################################

# Semilla con la que se han analizado los resultados
random.seed(9999)

# Cargamos los datos (sheet Raw Data nos es más cómodo que Data)
data = pd.read_excel("data/CTG.xls", "Raw Data")

# Eliminamos las 3 últimas filas que solo contienen valores de máximos y mínimos
data = data[:-3]

# Eliminamos la primera fila que está vacía
data = data[1:]

# Eliminamos las columnas que no contienen información relevante para la 
# clasificación
removed_columns = ["FileName", "Date", "SegFile", "b", "e", "A", "B", "C", "D",
                    "AD", "DE", "LD", "FS", "SUSP", "CLASS"]
data = data.drop(columns=removed_columns)

print("Datos sin normalizar:")
print(data)

# Separamos las etiquetas
labels = data.pop("NSP")
labels = pd.Categorical(labels)
labels = labels.rename_categories(["N", "S", "P"])
labels_names = ["Normal", "Suspect", "Pathologic"]

# Normalizamos los datos (estandarización)
data_norm = Normalizer().fit(data).transform(data)
data_norm = pd.DataFrame(data_norm)

print("------------------------------------------------------------------")
print("Datos normalizados:")
print(data)

# Nos quedan 24 clasificadores con 2126 instancias

# Aunque tenemos demasiadas características, podemos ver algunos datos estadísticos
print(data.describe())

# Tenemos un problema desbalanceado, muchas etiquetas de valor N
print(labels.describe())

################################################################################
# Visualizaciones
################################################################################

# ------------------------------------------------------------------------------
# Boxplots (usamos los datos normalizados para que estén en el mismo rango)
sns.boxplot(x="variable", y="value", data=pd.melt(data_norm))
plt.savefig("figure1.png")

# Se aprecian distribuciones bastante diferentes en las variables

# ------------------------------------------------------------------------------
# Correlación (https://seaborn.pydata.org/examples/many_pairwise_correlations.html)
sns.set_theme(style="white")

corr = data.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5)
plt.savefig("figure2.png")

# Hay algunas variables correladass negativamente, pero con correlación fuerte pocas

# ------------------------------------------------------------------------------
# Scatterplot

# De 2 variables correladas (puesto que esta es alta, sería recomendable quitar una)
sns.scatterplot(data=data, x="Min", y="Width")
plt.savefig("figure3.png")

################################################################################
# Preprocesamiento
################################################################################

# Como tenemos bastantes datos, separamos un 20% para test
# Necesitamos que los datos estén normalizados antes de aplicar PCA
x_train, x_test, y_train, y_test = train_test_split(data_norm, labels, test_size=0.2)


# Calculamos PCA sobre train
n_components = 5    # Bajo porque los tres modelos que tenemos tienden a 
                    # sobreajustar
pca = PCA(n_components=n_components, svd_solver="randomized", 
            whiten=True).fit(x_train) # No queremos que las componentes devueltas
                                      # estén correladas

# Aplicamos PCA
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

# ------------------------------------------------------------------------------
# Mostramos algunos gráficos de las componentes

df = pd.DataFrame(x_train_pca)
df["Labels"] = y_train

sns.pairplot(df, hue="Labels").fig.suptitle("Componentes de PCA con sus verdaderas etiquetas", y=1.01)
plt.savefig("figure4.png")

################################################################################
# Clasificando con SVM
################################################################################
print("------------------------------------------------------------------")
print("Clasificando con SVM")
print("------------------------------------------------------------------")

# Declaración de hiperparámetros
param_grid = {
    "C": [10, 1e3, 1e5],        # Valores de regularización
    "gamma": [0.01, 0.001],     # Grado de influencia de un valor
    "kernel": ["rbf", "poly"]   # Kernel rbf o polinomial
}

# Test de hiperparámetros con cross-validation de 5 folds
clf = GridSearchCV(SVC(class_weight="balanced"),
                    param_grid)
clf = clf.fit(x_train_pca, y_train)

print("Mejores hiperparámetros del modelo:")
print(clf.best_params_)
print("\nMejor score obtenido:")
print(clf.best_score_)
# Se nos queda un modelo con alto grado de regularización y kernel polinomial

# Guardamos el mejor estimador
best_svm = clf.best_estimator_

#############################
# Evaluación final del modelo
#############################

# Predecimos sobre los datos de test que habíamos reservado
y_pred = clf.predict(x_test_pca)

print("\nResultados de la predicción sobre test:")
print(classification_report(y_test, y_pred, target_names=labels_names))
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------------------------------------------------------
# Mostramos gráficos sobre los resultados
disp = plot_confusion_matrix(best_svm, x_test_pca, y_test,
                             display_labels=labels_names, cmap=plt.cm.Blues)
plt.savefig("figure1.png")

# Clasificación realizada en test
df = pd.DataFrame(x_test_pca)
df["Labels"] = y_test
df["Pred"] = y_pred

sns.pairplot(df, hue="Labels").fig.suptitle("Componentes de PCA con sus verdaderas etiquetas", y=1.01)
plt.savefig("figure5.png")

sns.pairplot(df, hue="Pred").fig.suptitle("Componentes de PCA con etiquetas predichas", y=1.01)
plt.savefig("figure6.png")

# Tenemos buenos resultados, pero no es una gran mejora respecto respecto de una
# predicción sin aprendizaje (si siempre dijéramos N acertaríamos un 77% de las veces)