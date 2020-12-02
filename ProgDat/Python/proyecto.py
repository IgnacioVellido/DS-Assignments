#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Cardiotocogram Dataset
Ignacio Vellido Expósito
================================================================================
3 métodos de predicción + PCA (preprocesado)
(poner alguna gráfica también)

Data Set Information:

2126 fetal cardiotocograms (CTGs) were automatically processed and the respective 
diagnostic features measured. The CTGs were also classified by three expert 
obstetricians and a consensus classification label assigned to each of them. 
Classification was both with respect to a morphologic pattern (A, B, C. ...) 
and to a fetal state (N, S, P). Therefore the dataset can be used either for 
10-class or 3-class experiments.

Exam data		
    FileName	of CTG examination		
    Date	of the examination		
    b	start instant		
    e	end instant		
Measurements		
    LBE	baseline value (medical expert)		
    LB	baseline value (SisPorto)		
    AC	accelerations (SisPorto)		
    FM	foetal movement (SisPorto)		
    UC	uterine contractions (SisPorto)		
    ASTV	percentage of time with abnormal short term variability  (SisPorto)		
    mSTV	mean value of short term variability  (SisPorto)		
    ALTV	percentage of time with abnormal long term variability  (SisPorto)		
    mLTV	mean value of long term variability  (SisPorto)		
    DL	light decelerations		
    DS	severe decelerations		
    DP	prolongued decelerations		
    DR	repetitive decelerations		
    Width	histogram width		
    Min	low freq. of the histogram		
    Max	high freq. of the histogram		
    Nmax	number of histogram peaks		
    Nzeros	number of histogram zeros		
    Mode	histogram mode		
    Mean	histogram mean		
    Median	histogram median		
    Variance	histogram variance		
    Tendency	histogram tendency: -1=left assymetric; 0=symmetric; 1=right assymetric		
Classification		
    A	calm sleep		
    B	REM sleep		
    C	calm vigilance		
    D	active vigilance		
    SH	shift pattern (A or Susp with shifts)		
    AD	accelerative/decelerative pattern (stress situation)		
    DE	decelerative pattern (vagal stimulation)		
    LD	largely decelerative pattern		
    FS	flat-sinusoidal pattern (pathological state)		
    SUSP	suspect pattern		
    CLASS	Class code (1 to 10) for classes A to SUSP		
    NSP	Normal=1; Suspect=2; Pathologic=3		


Vamos a aplicar clasificación respecto a la última variable (NSP), el resto de 
columnas de clasificación las quitaremos.
También eliminaremos la columnas con información del exámen médico por no ser
medidas usadas en la clasificación.
"""

################################################################################
# Librerías
################################################################################

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Preprocesamiento
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

# Cargamos los datos (Raw Data nos es más cómodo que Data)
data = pd.read_excel("data/CTG.xls", "Raw Data")

# Eliminamos las 3 últimas filas que solo contienen valores de máximos y mínimos
data = data[:-3]

# Eliminamos las primera fila que está vacía
data = data[1:]

# Eliminamos la primeras dos columnas pues no contienen información relevante
# para la clasificación
removed_columns = ["FileName", "Date", "SegFile", "b", "e", "A", "B", "C", "D",
                    "AD", "DE", "LD", "FS", "SUSP", "CLASS"]
data = data.drop(columns=removed_columns)

print("Data:")
print(data)

labels = data.pop("NSP")
labels_names = ["Normal", "Suspect", "Pathologic"]

################################################################################
# Preprocesamiento
################################################################################

# Como tenemos bastantes datos, separamos un 20% para test
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Calculamos PCA sobre train
n_components = 5    # Bajo porque los tres modelos que tenemos tienden a 
                    # sobreajustar
pca = PCA(n_components=n_components, svd_solver="randomized", 
            whiten=True).fit(x_train) # No queremos variables correladas ?

# Aplicamos PCA
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

# ------------------------------------------------------------------------------
# Mostramos algunos gráficos

df = pd.DataFrame(x_test_pca)
df["Labels"] = pd.factorize(y_train)

sns.pairplot(df, hue="Labels").savefig("pair.png")

# ################################################################################
# # Clasificando con SVM
# ################################################################################
# print("------------------------------------------------------------------")
# print("Clasificando con SVM")
# print("------------------------------------------------------------------")

# # Declaración de hiperparámetros
# param_grid = {
#     "C": [10, 1e3, 1e5],
#     "gamma": [0.01, 0.001],
#     "kernel": ["rbf", "poly"]
# }

# # Test de hiperparámetros con cross-validation de 5 folds
# clf = GridSearchCV(SVC(class_weight="balanced"), 
#                     param_grid)
# clf = clf.fit(x_train_pca, y_train)

# print("Mejores hiperparámetros del modelo:")
# print(clf.best_params_)
# print("\nMejor score obtenido:")
# print(clf.best_score_)

# # Guardamos el mejor estimador
# best_svm = clf.best_estimator_

# #############################
# # Evaluación final del modelo
# #############################

# y_pred = clf.predict(x_test_pca)

# print("\nResultados de la predicción sobre test:")
# print(classification_report(y_test, y_pred, target_names=labels_names))
# print("Matriz de confusión:")
# print(confusion_matrix(y_test, y_pred))

# # ------------------------------------------------------------------------------
# # Mostramos gráficos sobre los resultados
# disp = plot_confusion_matrix(best_svm, x_test_pca, y_test,
#                                 display_labels=labels_names, cmap=plt.cm.Blues)
# # plt.show()
# plt.savefig("prueba.png")

# ################################################################################
# # Clasificando con KNN
# ################################################################################
# print("------------------------------------------------------------------")
# print("Clasificando con SVM")
# print("------------------------------------------------------------------")

# # Declaración de hiperparámetros
# param_grid = {

# }

# # Test de hiperparámetros
# clf = GridSearchCV(SVC(kernel="", class_weight=""), param_grid)
# clf = clf.fit(x_train_pca, y_train)

# print("Mejores hiperparámetros del modelo:")
# print(clf.best_estimator_)

# # ------------------------------------------------------------------------------
# # Fit con CV
# # kf = KFold(n_splits=10, shuffle=True)

# for tr, tst in kf(x_train_pca):
#     # Cogemos los datos de este kfold
#     x_tr = x_train_pca[tr,:]
#     y_tr = y_train[tr,:]
#     x_tst = x_test_pca[tst,:]
#     y_tst = y_test[tst,:]

#     # Lanzamos el algoritmo

#     # Mostramos la métrica sobre conjunto de validación

# #############################
# # Evaluación final del modelo
# #############################


# ################################################################################
# # Clasificando con Random Forest
# ################################################################################
# print("------------------------------------------------------------------")
# print("Clasificando con SVM")
# print("------------------------------------------------------------------")


# # Declaración de hiperparámetros
# param_grid = {

# }

# # Test de hiperparámetros
# clf = GridSearchCV(SVC(kernel="", class_weight=""), param_grid)
# clf = clf.fit(x_train_pca, y_train)

# print("Mejores hiperparámetros del modelo:")
# print(clf.best_estimator_)

# # ------------------------------------------------------------------------------
# # Fit con CV
# # kf = KFold(n_splits=10, shuffle=True)

# for tr, tst in kf(x_train_pca):
#     # Cogemos los datos de este kfold
#     x_tr = x_train_pca[tr,:]
#     y_tr = y_train[tr,:]
#     x_tst = x_test_pca[tst,:]
#     y_tst = y_test[tst,:]

#     # Lanzamos el algoritmo

#     # Mostramos la métrica sobre conjunto de validación

# #############################
# # Evaluación final del modelo
# #############################