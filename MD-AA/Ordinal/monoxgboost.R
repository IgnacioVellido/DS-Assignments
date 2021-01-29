# Librerías
library(tidyverse)
library(xgboost)

# ------------------------------------------------------------------------------

# Leer datos
df <- read_csv("data/esl.arff", col_names = FALSE, skip = 43)

label_name <- colnames(df)[ncol(df)]
df$labels <- df[[label_name]]
df[[label_name]] <- NULL

# ------------------------------------------------------------------------------

# Función para clasificar K-1 modelos
classify <- function(df) {
  # Hacer particiones OVA tipo (if label > i -> 1, else -> 0)
  labels <- as.integer(unique(df$labels))
  num_part <- length(labels) - 1
  
  partitions <- vector("list", length(num_part))
  
  for(i in 1:num_part) {
    partitions[[i]] <- df
    partitions[[i]] <- partitions[[i]] %>% mutate(labels = ifelse(labels <= i, 0, 1))
    
    # Para realizar el entrenamiento, necesitamos que las etiquetas se codifiquen
    # como factores
    partitions_labels[[i]] <- partitions[[i]]$labels %>% as.factor()
    partitions[[i]]$labels <- NULL
  }
  
  # Aplicar xgboost
  models <- vector("list", length(num_part))
  for(i in 1:num_part) {
    models[[i]] <- xgboost(partitions[[i]] %>% as.matrix(), partitions_labels[[i]],
                           nrounds = 1, monotone_constraints=1)
  }
  
  # Devolver modelos
  models
}

# ------------------------------------------------------------------------------

# Función para calcular probabilidades reales y hacer predicciones
make_predictions <- function(models, df) {
  # Calcular probabilidades del df para un modelo
  pr_list <- lapply(models,
                    function(m,r) {
                      # Solo nos interesa la segunda probabilidad, que sea > i
                      predict(m,r) %>% as.data.frame() -> x
                      x %>% mutate(row = rownames(x))
                    },
                    df)
  
  
  # Juntar probabilidades por filas del df
  pr_list <- reduce(pr_list, left_join, by="row")
  
  # Eliminamos columna auxilar
  pr_list$row <- NULL
  
  # Pertenencia a una clase o no (grado alto)
  pr_list <- pr_list > 0.8
  
  
  # Fórmula del clasificador multiclase
  pr_list %>% apply(1, function(x) {
    1 + sum(x)
  }) %>% as.data.frame()
}

# ------------------------------------------------------------------------------

models <- classify(df)
test <- df %>% select(-labels) %>% as.matrix()
pred <- make_predictions(models, test)

pred

# ------------------------------------------------------------------------------

# Ver medidas de aciertos
f1_score <- function(predicted, expected, positive.class="1") {
  res <- list()
  
  cm = as.matrix(table(expected, predicted))
  res$cm <- cm
  
  tp <- diag(cm)
  fp <- cm[lower.tri(cm)]
  fn <- cm[upper.tri(cm)]
  
  res$precision <- tp / (tp + fp)
  res$recall <- tp / (tp + fn)
  
  
  f1 <-  ifelse(res$precision + res$recall == 0, 0,
                2 * res$precision * res$recall / (res$precision + res$recall))
  
  #Assuming that F1 is zero when it's not possible compute it
  f1[is.na(f1)] <- 0
  res$precision[is.na(res$precision)] <- 0
  res$recall[is.na(res$recall)] <- 0
  
  #Binary F1 or Multi-class macro-averaged F1
  res$f1 <- ifelse(nlevels(expected) == 2, f1[positive.class], mean(f1))
  
  res
}

pred$class <- factor(x = pred$., levels = df$labels %>% as.factor() %>% levels())

f1_score(pred$class, df$labels %>% as.factor())

# ------------------------------------------------------------------------------

# Vemos que los resultados obtenidos son mayormente buenos, donde los fallos
# cometidos solo se dan una o dos clases arriba o abajo.

# Notamos que la predicción de las clases extremas (la 1 y la 9) no es en ningún
# momento correcta. Es más, las probabilidades de estas clases son tan bajas en
# cada caso que nunca se predicen.