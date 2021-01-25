# Librerías
library(tidyverse)
library(tree)

# Leer datos
df <- read_csv("data/esl.arff", col_names = FALSE, skip = 43)

# Cada columna es categórica ordinal
for (i in 1:(ncol(df)-1)) {
  df[[i]] <- df[[i]] %>% as.factor()
}

# Añadir nombre de columna a las etiquetas
colnames(df)[5] <- "labels"

# Ordenar df por etiquetas
df <- df %>% arrange(labels)


# Función para clasificar K-1 modelos
classify <- function(df) {
  # Realizar particiones
  labels <- as.integer(unique(df$labels))
  num_part <- length(labels) - 1
  
  partitions <- vector("list", length(num_part))
  
  for(i in 1:num_part) {
    partitions[[i]] <- df
    partitions[[i]] <- partitions[[i]] %>% mutate(labels = ifelse(labels <= i, 0, 1))
    
    # Para realizar el entrenamiento, necesitamos que las etiquetas se codifiquen
    # como factores
    partitions[[i]]$labels <- partitions[[i]]$labels %>% as.factor()
  }
  
  # Clasificar
  models <- vector("list", length(num_part))
  for (i in 1:num_part) {
    models[[i]] <- tree(labels ~ ., data = partitions[[i]])
  }
  
  # Devolver modelos
  models
}

# Función para calcular probabilidades reales y hacer predicciones
make_predictions <- function(models, df) {
  # Calcular probabilidades del df para un modelo
  pr_list <- lapply(models,
                    function(m,r) {
                      # Solo nos interesa la segunda probabilidad, que sea > i
                      predict(m,r)[,2] %>% as.data.frame() -> x
                      x %>% mutate(row = rownames(x))
                    },
                    df)
  
  
  # Juntar probabilidades por filas del df
  pr_list <- reduce(pr_list, left_join, by="row")
  
  
  # Mover la columna auxiliar row a la primera posición
  # y llenarla con unos, preparando el cálculo de las
  # probabilidades de cada clase
  pr_list <- pr_list[, c(2,1,3:ncol(pr_list))]
  pr_list$row <- 1
  
  
  # Calcular probabilidades reales de cada clase
  apply(pr_list, 1, function(row) {
    probs <- vector("integer", length(row))
    
    for(i in 2:length(row)) {
      probs[i-1] <- row[[i-1]] * (1 - row[[i]])
    }
    
    # Probabilidad última clase
    probs[length(row)] <- row[[length(row)]]
    
    # Devolver probabilidad máxima (y el índice)
    m  <- max(probs)
    data.frame(class=which(probs == m), prob = m)
  }) %>%
    reduce(bind_rows) # Juntar en un solo dataframe
}


# Aplicar las funciones a nuestro dataset
models <- classify(df)
make_predictions(models, df %>% select(-labels))

df %>% head()