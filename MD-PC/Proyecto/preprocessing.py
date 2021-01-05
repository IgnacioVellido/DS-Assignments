""" 
Preprocesamiento
- No hace falta estandarizar, no estaría tampoco mal
- Agrupar las binarias (has..) en grupos
  - has_secondary_use (hay que añadir más para las combinaciones)
  - has_superstructure
- geo_level3 o 2 quizás se puedan agrupar en intervalos
- Pasar las categóricas a números ? En sí las letras no tienen significado
- Quizás agrupar la edad en intervalos
- Ver qué hacer con las duplicadas
- Los árboles son robustos al ruido, así que las variables extremadamente 
desbalanceadas no deberían hacer mucho daño. Considerar si quitar o no según el
coste computacional que haya en principio.
- Age de 995 mantenerlo, a ver si el árbol consigue algo del estilo if age>200 ...

- Quizá la distancia, al ser datos del mismo terremoto sea muy relevante. 
Se podría reflejar en una combinación de las has_structure y geom_ids 
(el epicentro estaría por unos valores concretos de geom_id)
"""
################################################################################
# Libraries
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
from sklearn.neighbors import KNeighborsClassifier

################################################################################
# Load data
################################################################################

train_values = pd.read_csv("data/train_values.csv")
test_values  = pd.read_csv("data/test_values.csv")
train_labels = pd.read_csv("data/train_labels.csv")

preprocessed_train = train_values
preprocessed_test = test_values

################################################################################
# Write data
################################################################################

# preprocessed_train.to_csv("data/preprocessed_train.csv")
# preprocessed_test.to_csv("data/preprocessed_test.csv")


















"""
Deshaciendo variables dummy
```{r}
MultChoiceCondense<-function(vars,indata){
  tempvar<-matrix(NaN,ncol=1,nrow=length(indata[,1]))
  dat<-indata[,vars]
  for (i in 1:length(vars)){
    for (j in 1:length(indata[,1])){
      if (dat[j,i]==1) tempvar[j]=i
    }
  }
  return(tempvar)
}

condense <- 
  c("has_superstructure_adobe_mud",
    "has_superstructure_mud_mortar_stone",
    "has_superstructure_stone_flag",
    "has_superstructure_cement_mortar_stone",
    "has_superstructure_mud_mortar_brick",
    "has_superstructure_cement_mortar_brick",
    "has_superstructure_timber",
    "has_superstructure_bamboo",
    "has_superstructure_rc_non_engineered",
    "has_superstructure_rc_engineered",
    "has_superstructure_other"
    )
train_values$has_superstructure <- MultChoiceCondense(condense, train_values)

condense <- 
  c("has_secondary_use",
    "has_secondary_use_agriculture",
    "has_secondary_use_hotel",
    "has_secondary_use_rental",
    "has_secondary_use_institution",
    "has_secondary_use_school",
    "has_secondary_use_industry",
    "has_secondary_use_health_post",
    "has_secondary_use_gov_office",
    "has_secondary_use_use_police",
    "has_secondary_use_other"
  )
train_values$has_secondary_use <- MultChoiceCondense(condense, train_values)

train_values$has_superstructure %>% head(100) %>% as.data.frame()
# train_values$has_secondary_use_agriculture %>% head(100)
train_values %>% head()
"""
