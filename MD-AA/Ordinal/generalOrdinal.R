# Librerías
library(tidyverse)
library(tree)

# Leer datos
df <- read_csv

classify <- function(df) {
    num_part <- # Number of partitions

    # Hacer particiones
    partitions <- vector("list", length(num_part))


    # Clasificar
    models <- vector("list", length(num_part))
    for (i in 1:num_part) {
        models[i] <- tree(labels ~ ., data = partitions[i])
    }

    # Devolver modelos
    models
}

predict <- function(models, df) {
    for(row in df) {
        # Predecir y recuperar probabilidades para cada modelo
        prob <- lapply(models, function(m, row) {
            predict(m, row, type="probability")
        }, row)

        # Calcular clase definitiva
        # 

        # Guardar resultados
    }
}