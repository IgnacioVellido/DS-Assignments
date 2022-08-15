# Kaggle Competition: Usos del suelo con Sentinel 2

Using multi-input NN (images + coordinates)

## Overview
### Description
El objetivo de esta competición es diseñar un sistema automático que determine la clase de usos del suelo a partir de una imagen aérea o de satélite. Es decir, dada una imagen, queremos desarrollar un modelo de clasificación basado en redes neuronales que determine la clase correcta de dicha imagen.

### Evaluation

## Data Description
El objetivo de esta tarea es desarrollar un modelo de redes neuronales artificiales capaz de identificar el uso del suelo a partir de una imagen RGB. Utilizaremos una base de datos que contiene 26 classes. Se proporcionan dos ficheros zip, uno de entrenamiento (etiquetado de acuerdo a la carpeta donde se encuentran las imágenes) y otro de test (todos mezclados). Utilizad "LULC_100samples2021" para ajustar el modelo y luego enviad un fichero CSV de test (de acuerdo al "sample" proporcionado) para chequear lo bueno que es vuestro clasificador. 

Descripción de los archivos:
- train.zip - el conjunto de entrenamiento dividido en 26 carpetas, cada carpeta corresponde a un tipo de uso de suelo 
- test.zip - el conjunto de test. Todas las imágenes están mezcladas 
- sample.csv - ejemplo del csv que hay que enviar

## Ideas:
- Las coordenadas van codificadas en el nombre del archivo. Podría ser útil extraerlas y hacer una red con 2 tipos de entradas (Habría que preguntar si los archivos de test también lo tienen).
  - Suponiendo que se puede, se podría entrenar las coordenadas con NN y concatenar (https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/), o entrenarlas con RF/SVM... y hacer una selección por probabilidades.
