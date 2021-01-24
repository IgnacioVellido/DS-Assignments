# Presentación

- Variables extremadamente desbalanceadas (más del 90%). Poca información nueva aportada por ellas, pero en parte mayor facilidad para discriminar con árboles. Además, los árboles son robustos al ruido (u hojas con poca información contenida) puesto que los métodos de poda las eliminarán.
    - Sin distribuciones normales
   - Correlación solo en dos columnas (Altura y Nº de plantas)
   - Clases desbalanceadas, la de grado 1 con menor proporción, 2 como mayoritaria.
   - ~16mil instancias duplicadas, pero de edificios diferentes. Es posible que sean de una misma urbanización/zona de una ciudad. Sorprendía que se repitieran por el count_families, pero la mayoría son de 1 sola (posiblemente sean casas).
   - Puede que haya valores perdidos no codificados como NA.
   - Los IDs de posición geográfica deberían ser relevantes para el problema (un solo terremoto y por tanto un solo epicentro). Ahora bien, suponiendo que IDs consecutivos son zonas cercanas, no se aprecia relación ni orden alguno.
   - La información geográfica de geo_level 2 y 3 está codificada de manera resumida en geo_level_1. Merece la pena por tanto quitarse esas dos puesto que los algoritmos no aceptan variables categóricas con tan cantidad de categorías.
   - plan_configuration, legal_ownership_status están muy desbalanceadas pero cada categoría contiene etiquetas de cada una de las clases, por lo que no sirven para determinar una de ellas.

1. Introducción
   1. Metodología (mini gráfico)
      1. EDA
      2. Preprocesamiento
      3. Entrenamiento
      4. Evaluación
      5. Revisión y repetición
   2. Datos destacados del dataset
      1. Dataset desbalanceado
      2. Variables muy desbalanceadas, sin ayudar a la clasificación
      3. Instancias duplicadas
2. Preprocesamientos innecesarios
   1. Normalización influyente al no ser un algoritmo paramétrico
   2. Discretización mediante medidas incluída en los propios algoritmos
      1. Peligro de elegir malos intervalos
      2. Posible influencia de valores extremos (???)
   3. Datos limpios
      1. Sin valores nulos
      2. Algunos extremos en ciertas variables, no muy influyentes en árboles
      3. Ordenados
      4. Posible agrupación de algunas variables binarias, que por la codificación de los algoritmos en R no aceptaba (acababan con demasiadas categorías)
3. Desbalanceo de clases (y duplicidad), efectos en la predicción con árboles
   1. Hablar también de la duplicidad
4. Algoritmos de aprendizaje usados
   1. Algoritmo basado en CART con medida GINI, poda a posteriori forzando una profundidad mínima
   2. Algoritmo basado en C4.5 con medida de entropía, poda a posteriori dentro del algoritmo
5. Tabla de submissions
6. Gráfica de rankings (? 639, 773, depende del día) y accuracy, en val/CV también
   1. Organizar por método, y dentro por técnicas
   2. Número total de submissions por algoritmo
7. Técnicas de preprocesamiento utilizadas
   1. Undersampling/Noise Reduction
      1. TomekLinks
      2. ENN
      3. CNN
      4. Random Undersampling
   2. Oversampling
      1. SMOTE
      2. SMOTE + ENN
   3. Discretización con KMeans en 2 variables continuas (20 y 10 particiones)
   4. Selección de características (tamaños 10-15-40-80, codificación one-hot bastante influyente en la información representada)
      1. Manual
         1. geo2y3 por verse incluídas en geo1 y no aportar información especial
         2. Correladas
         3. Totalmente desbalanceadas (más del 95%, sin la clase minoritaria ayudando a clasificar)
      2. Mediante algoritmos
         1. KBest con distancia chi2
         2. KBest con mutual information (MI): Dependencia entre variables
   5. (Y muchas más...) Combinaciones de las anteriores (no incluir esto de abajo, solo decirlo)
      1. Selección + Undersampling
      2. Selección + Oversampling
      3. Discretización + Undersampling
8. Modelo vencedor
9. Justificaciones del modelo vencedor
10. Conclusiones
   1. Problemas
      1. Entrenamientos rápidos con números moderados de instancias y/o características, lentos al aumentar
         1. Lentitud de algoritmos de preprocesamiento (basados en cálculos de distancias: ENN, Tomek...). Selección de características prácticamente obligatoria (vienen de onehot)
      2. Falta de memoria
      3. Sesgo en árboles pequeños
      4. Facilidad de overfitting
   2. CART (Ponerlo en contraposición de C4.5, a ambos lados (rojo, verdad))
      1. Árboles pequeños con mucha capacidad de generalización (verde)
      2. Dificultades con datasets desbalanceados (rojo)
      3. (Implementación en R) relativamente lento (rojo)
   3. C4.5
      1. Árboles grandes con facilidad de overfitting (rojo)
      2. Una mayor profundidad se adapta mejor al desbalanceo (verde)
      3. (Implementación en R) muy rápido (verde)

<!-- - Una diapositiva final con la evolución del ranking y de la tasa de acierto asociado
a cada entrega en DrivenData, en formato gráfico. El valor de ranking y tasa de
acierto final conseguido, junto a número de subidas final que se han realizado a
DrivenData.
- Una diapositiva con la lista completa de algoritmos de preprocesamiento utilizados
y sus diferentes configuraciones a lo largo de la práctica (qué se ha probado en el
transcurso de la misma con intervalos de parámetros). Además, también se considerarán
las diferentes configuraciones (parámetros) del algoritmo de aprendizaje
- Una diapositiva con el detalle de la estrategia que ha obtenido el mejor ranking
(pipeline de técnicas) y su justificación

El resto de diapositivas se puede utilizar para describir el proceso y detallar mejor
los resultados, motivaciones, justificaciones, decisiones tomadas, etc. No se recomienda
la inclusión de código en la presentación ni descripciones teóricas de los algoritmos explicados
en clase. Sí que será necesario describir aquellas técnicas utilizadas que no se
hayan visto en clase (breve descripción), incluir referencias, etc... -->

# ######################

prune.missclass usa el classification error rate

Gráficos:
- Piechart desbalanceo, pre y post procesamiento
- Tamaños de árboles (o mejor tabla)
- Progressión de F1, val y test (una para cada algoritmo)
- Violinplot ?

El árbol CART inicial sobre todas las variables nos decía que los predictores más importantes era geo_1, num_floors ... (revisar)

Tal y como se describe más adelante, la creación de las ramificaciones de los árboles se consigue mediante el algoritmo de recursive binary splitting. Este algoritmo identifica y evalúa las posibles divisiones de cada predictor acorde a una determinada medida (RSS, Gini, entropía…). Los predictores continuos tienen mayor probabilidad de contener, solo por azar, algún punto de corte óptimo, por lo que suelen verse favorecidos en la creación de los árboles.

TODO:
- Ponderización en evaluate_CV
- SMOTE tras/pre reducción de variables y eliminación de ruido
- Random Undersampling clase 2 (si va bien con el dataset sin tocar, esto no debería ser apropiado)

# Metodología

1. Realización de un EDA para descubrir información de interés. Conclusiones (entre otras):
    - Variables extremadamente desbalanceadas (más del 90%). Poca información nueva aportada por ellas, pero en parte mayor facilidad para discriminar con árboles. Además, los árboles son robustos al ruido (u hojas con poca información contenida) puesto que los métodos de poda las eliminarán.
    - Sin distribuciones normales
   - Correlación solo en dos columnas (Altura y Nº de plantas)
   - Clases desbalanceadas, la de grado 1 con menor proporción, 2 como mayoritaria.
   - ~16mil instancias duplicadas, pero de edificios diferentes. Es posible que sean de una misma urbanización/zona de una ciudad. Sorprendía que se repitieran por el count_families, pero la mayoría son de 1 sola (posiblemente sean casas).
   - Puede que haya valores perdidos no codificados como NA.
   - Los IDs de posición geográfica deberían ser relevantes para el problema (un solo terremoto y por tanto un solo epicentro). Ahora bien, suponiendo que IDs consecutivos son zonas cercanas, no se aprecia relación ni orden alguno.
   - La información geográfica de geo_level 2 y 3 está codificada de manera resumida en geo_level_1. Merece la pena por tanto quitarse esas dos puesto que los algoritmos no aceptan variables categóricas con tan cantidad de categorías.
   - plan_configuration, legal_ownership_status están muy desbalanceadas pero cada categoría contiene etiquetas de cada una de las clases, por lo que no sirven para determinar una de ellas.

2. Aplicación de métodos de preprocesamiento en base a las ideas descubiertas.
3. Construcción de árboles, ajustando parámetros en aquellos que tengan y considerando diferentes tamaños de la partición de validación.
4. Evaluación de los modelos mediante CV y calculando el valor F1 micro para validación.
5. En caso de obtener resultados aceptables en el punto 4, calcular etiquetas del conjunto de test y evaluar en DrivenData.
6. Obtener conclusiones y repetir desde paso 2.

# Versiones

1. Agrupación de características binarias en dos subconjuntos (has_secondary_use y has_superstructure). De esta manera no se pierde información alguna y mantenemos la misma expresividad en los árboles. Puesto que se cuentan con muchas instancias, separación de un 20% para evaluación local.
   - Algunas variables categóricas cuentan con demasiadas categorías para CART.
   - Coste computacional alto en CART.

2. Se reducen características correladas (categorizando num_floors) y aquellas con demasiadas categorías (geo2, geo3, has_secondary_use). Se aumenta el conjunto de validación al 30%. Se eliminan algunas variables demasiado desbalanceadas (plan_configuration, legal_ownership_status). Se quitan los duplicados del dataset resultante.
   - Coste computacional alto. El dataset resultante sigue contando con unas 125.000 instancias.

3. Reducción de características a 10 con cálculo de scores en base a chi2 y reducción de instancias con ENN (one-hot-encoding aplicado previamente para el úso de estos métodos). Se mantienen fuera las variables eliminadas anteriormente, pero se mantienen los duplicados.
   
   Por un lado, queremos mantener los duplicados pues afecta a las medidas de GINI y entropía, de esta manera la información repetida se ve reflejada en los árboles pues se favorece la correcta etiquetación de estos. Por otro lado, al tener tan cantidad de instancias duplicadas podemos perder expresividad en árboles al contar con tantas columnas (y en unas pruebas iniciales se veía que sí, no se etiquetaba nunca la clase 1). Este problema es más frecuente en CART que en C4.5. También pueden generar overfitting en C4.5 si el método de poda coge umbrales bajos.

    El mantenerlos nos obliga a tener en cuenta además que al seleccionar características aumenta el número de duplicados en el dataset, y que se pueden generar instancias contradictorias (mismos valores, distinta clase de salida).

   - Resultados aceptables en CART, aunque se genera un árbol demasiado pequeño (por tanto no se aplica poda). 
   - Posible overfitting en C4.5, se produce un muy grande tras la poda.
   - Perdemos la proporción de etiquetas en nuestros datos, y podemos considerarlo tanto como algo bueno como algo malo. Para CART se da posiblemente bastante preferencia a la etiqueta 1.

4. Se parte del punto 3 y se aumenta el número de características a 15.
   - Se considera utilizar geo1 entero, pero no somos el experto, por lo que tomar la decisión en base a nuestra suposición a lo mejor no es buena idea.
    - Resultados malísimos en C4.5 probablemente causando overfitting por estos duplicados.
    - Se decide llevar preprocesamientos diferentes para cada algoritmo. Cuando obtenemos demasiadas instancias, el método "tree" de R falla.

5. Idem. pero con mutual_info_classif
   - Se prueba con diferentes tamaños de particiones, nos quedamos con 80%.
     - train 15%  = CV accuracy 62.61%, 0.6571 F1
     - train 50%  = CV accuracy 63.68%, 0.6638 F1
     - train 80%  = CV accuracy 64.34%, 0.6686 F1
     - train 100% = CV accuracy 64.67%
   - Parece que C4.5 funciona mejor con más datos y variables, probablemente porque hace proda tras conseguir el árbol. CART por el contrario no, se queda con un árbol pequeño así que la selección de instancias/características debe ser buena.
 
6. Punto 5 aumentando el conjunto de entrenamiento al 95%
   - Posible problema: Vienen de un one hot por lo que la reducción es enorme

7. Oversampling con método SMOTE seguido de un undersampling con ENN
   - Se desbalancea la clase mayoritaria, sobregeneralización en CART -> __Problema importante__
   - Buenos resultados en validación (del 5%), pobres en test.
   - Posible sobrepredicción las clases originalmente minoritarias.

8. Intentamos paliar el desbalanceo usando solo oversampling con SMOTE, aplicando one-hot. 300.000 instancias en total, separación del 35% para validación.

9.  Solo oversampling con SMOTE, sin aplicar one-hot
    - Se predice de manera excesiva la etiqueta 3.
    - Concluímos que el balanceo no funciona bien para el método C4.5, no descarta un efecto beneficioso en CART, puesto que genera árboles muy pequeños.

10. Pensando en v6: Discretización de variables continuas. Algo bueno en los árboles es que pueden trabajar con variables continuas, de manera que las medidas de GINI/entropía ayuden a elegir los puntos de separación en la partición de las ramas. Aún así, quizás ayudando previamente haciendo particiones de manera inteligente puede mejorar la calidad del árbol.
    - Sin reducción de características CART tarda mucho y agota máxima profundidad 
    - C4.5 va bien con todas, intentamos reducirle variables innecesarias

11. v10 con selección de características a 15, imitando a v6
   - Con 15 salen árboles muy pequeños, el uso las variables continuas es lo que iba bien. Pues también nos da errores de validación y CV muy malos
   - Se sube a 40, sigue con valores reguleros (CV 60.8112, f1 val 0.5742824, hojas 543, size 103180)
   - Se acaba evaluando en test con 80 características

12. v11 Reduciendo puntos con Tomek (sin eliminar duplicados, se lo dejamos a tomek)
   - No se puede, tarda mucho

TEST NO TIENE COUNT_FAMILIES == 9

<!-- v8
[1] "Number of leaves: 10641\nSize of the tree: 19169\n".
$precision
        1         2         3 
0.8599427 0.8882590 0.7733822 

$recall
        1         2         3 
0.9203354 0.9548657 0.7183099 

v9 
[1] "Number of leaves: 12790\nSize of the tree: 21501\n"
predicted
expected     1     2     3
       1 22564  1916   686
       2  4148 13425  7645
       3  1520  4993 18779
$precision
        1         2         3 
0.8447140 0.8982937 0.7899630 

$recall
        1         2         3 
0.9217320 0.9513854 0.7106797 

Predice mucho la clase 3
c45_test
    1     2     3 
18273 30009 38586 -->


# Submissions

|            | Version |      | F1 Test |        | F1 Val  |        | Nº de hojas |        | Nº instancias | % Val       | CV % accuracy K=5 C4.5 |
|------------|---------|------|---------|--------|---------|--------|-------------|--------|---------------|-------------| ---------------------- |
| Submission | CART    | C4.5 | CART    | C4.5   | CART    | C4.5   | CART        | C4.5   |               | CART | C4.5 |
| 1          | 3       | 3    | 0.4959  | 0.3364 | 0.9066  | 0.9593 | 23          | 724    | 81.120        | 20   | 20   | 92.29                  |
| 2          | 4       | 4    | 0.5356  | 0.6581 | 0.8855  | 0.9553 | 42          | 1.387  | 79.920        | 20   | 20   | 90.81                  |
| 3          |         | 5    |         | 0.6778 | 0.8756  | 0.9425 | 25          | 571    | 74.973        | 20   | 20   | 89.18                  |
| 4          |         | 6    |         | 0.6836*| -       | 0.6814 | -           | 5.586  | 179.275       | -    |  5   | 64.85                  |
| 5          |         | 7    |         | 0.5064 |         | 0.8724 |             | 3.896  | 125.377       |      |  5   | 93.20                  |
| 6          |         | 8    |         | 0.5812 |         | 0.9328 |             | 10.641 | 302.703       |      | 35   | 72.17                  |
| 7          |         | 9    |         | 0.5703 |         | 0.8512 |             | 12.790 | 302.703       |      | 35   | 70.23                  |
| 8          |         | 10   |         | 0.6734 |         | 0.7018 |             | 4.821  | 108.611       |      |  5   | 62.25                  |
| 9          |         | 11   |         | 0.6285 |         | 0.6388 |             | 1.995  | 103.180       |      |      | 61.72                  |
| 10         |         |      |         |        |         |        |             |        |               |      |      |


Demasiado overfitting con oversampling, pero con undersampling no ha ido mal (v6 tenía under)

Empty cell = No submission

CART1 Y CART2 se tuvieron que repetir por errores en la construcción del árbol (C4.5 1 se repitió para ver que no había fallos, y da el mismo resultado)

# Conclusiones 

## Sobre los algoritmos de aprendizaje

Hemos usado J48 (basado en C4.5, con medida de entropía) y tree (basado en CART, árboles de partición binaria con criterio GINI).

CART 
- Árboles pequeños, muy sensibles al desbalanceo
- Método algoritmo que acarrea problemas de computación, quizás por aplicar particiones binarias. 
- Demasiada facilidad a la hora de generalizar para nuestro problema.
- No apto para problemas desbalanceados (COMPROBAR CON SMOTE + SELECCIÓN + DUPLICADOS)
  
C4.5 
- Árboles grandes, con pruning por defecto, no sensible al nº de instancias ni al desbalanceo. 
- Mayor facilidad de overfitting.
- Se adapta bien a problemas desbalanceados al generar un árbol grande.

## Sobre las técnicas de preprocesamiento

1. El nº de instancias es relevante no solo por la implicación computacional que acarrea, sino por redundancia y variación de información explicada y la facilidad de los algoritmos a adaptarse a ella.
2. No necesitamos normalizar los datos en árboles. Además, no hacerlo ayuda a interpretar los resultados.
3. Un oversampling no apropiado puede ser más peligroso que el propio dataset desbalanceado.

Se ha probado con:
- Sin selección de características
- Eliminación de características (principales) (El mejor)
- Selección de características (10 en total)
- Random instance selection, usando conjuntos como validación
- SMOTE
- SMOTE + ENN
- ENN mediante chi2
- ENN mediante multi_class

# Enlaces

- https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd
- https://scikit-learn.org/stable/modules/feature_selection.html
- https://cran.r-project.org/web/packages/stablelearner/stablelearner.pdf
- https://stats.stackexchange.com/questions/49226/how-to-interpret-f-measure-values
- https://machinelearningmastery.com/feature-selection-with-categorical-data/

Imbalance
- https://machinelearningmastery.com/undersampling-algorithms-for-imbalanced-classification/
- https://machinelearningmastery.com/multi-class-imbalanced-classification/

Imbalance in R
- https://www.rdocumentation.org/packages/unbalanced/versions/2.0 2015
- https://rdrr.io/cran/NoiseFiltersR/f/README.md 2016
- https://rdrr.io/cran/UBL/ 2017

Imbalance in Python
- https://github.com/scikit-learn-contrib/imbalanced-learn

Trees
- https://www.quora.com/What-are-the-differences-between-ID3-C4-5-and-CART
- https://stats.stackexchange.com/questions/28029/training-a-decision-tree-against-unbalanced-data
- https://weka.8497.n7.nabble.com/Producing-a-perfect-decision-tree-using-J48-td11751.html
- https://datascience.stackexchange.com/questions/43444/how-to-evaluate-feature-quality-for-decision-tree-model
- https://www.cienciadedatos.net/documentos/33_arboles_decision_random_forest_gradient_boosting_c50