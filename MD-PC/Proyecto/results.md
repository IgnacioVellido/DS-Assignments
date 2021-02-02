Particiones binarias creadas de arriba a abajo de manera greedy
Sobresimplificación en CART
Tocar el parámetro M para reducir la complejidad del árbol, haciendo que una hoja deba ser representativa de un buen grupo de instancias

Desbalanceo pero la medida F1 minor está basada en accuracy, lo que penaliza la minoritaria

Metodología
Partiendo de la información descubierta en el EDA se entra en un ciclo de preprocesamiento - entrenamiento - evaluación 

Submissions
En total 12 versiones del dataset diferentes, con 12 evaluaciones en test para el algoritmo C4.5 y 2 para CART

Técnicas de preprocesamiento utilizadas - 2
   Discretización: Se sabe que los propios algoritmos utilizan mecanismos para la elección de la mejor partición del las variables numéricas, pero se decide probar utilizando un método de clústering con diferentes tamaños de particiones
      Peligro de elegir malos intervalos
      Posible influencia de valores extremos al generar intervalos muy grandes

21: Se dejaba crecer el árbol CART al máximo y luego se buscaba la mejor poda con CV. Con C4.5 al tener el proceso de poda ya incluído solo se evaluaba.

23: Discretización y normalización.
Kmeans con clústers

24: v3 muy básica con preprocesamiento mínimo (con ruido, )
v8 y v9 es SMOTE con diferentes parámetros ?

27: The depth of the tree grows linearly with the number of variables, but the number of branches grows exponentially with the number of states.
Más difícil encontrar las particiones apropiadas

28: Fine tuning de los hiperparámetros

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
         1. Optimización del nivel de poda
      4. Evaluación
      5. Revisión y repetición
      6. Optimización del mejor model (nivel de poda e hiperparámetros)
   2. Datos destacados del dataset
      1. Dataset desbalanceado
      2. Variables muy desbalanceadas, sin ayudar a la clasificación
      3. Instancias duplicadas
2. Preprocesamientos innecesarios
   1. Normalización no influyente al no ser un algoritmo paramétrico
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
   1. Reducción manual de características
   2. One hot encoding
   3. Reducción de características con KBest y MI (nº final: 15)
   4. Eliminación de duplicados
   5. Reducción de instancias con ENN
   6. Selección de mejores hiperparámetros con Grid search
9.  Justificaciones del modelo vencedor
10. Conclusiones
   7. Problemas
      1. Entrenamientos rápidos con números moderados de instancias y/o características, lentos al aumentar
         1. Lentitud de algoritmos de preprocesamiento (basados en cálculos de distancias: ENN, Tomek...). Selección de características prácticamente obligatoria (vienen de onehot)
      2. Falta de memoria
      3. Sesgo en árboles pequeños
      4. Facilidad de overfitting
   8. CART (Ponerlo en contraposición de C4.5, a ambos lados (rojo, verdad))
      1. Árboles pequeños con mucha capacidad de generalización (verde)
      2. Dificultades con datasets desbalanceados (rojo)
      3. (Implementación en R) relativamente lento (rojo)
   9. C4.5
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
   - La información geográfica de geo_level 2 y 3 está codificada de manera resumida en geo_level_1. Merece la pena por tanto quitarse esas dos puesto que los algoritmos no aceptan variables categóricas con tal cantidad de categorías.
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
   - Parece que C4.5 funciona mejor con más datos y variables, probablemente porque hace poda tras conseguir el árbol. CART por el contrario no, se queda con un árbol pequeño así que la selección de instancias/características debe ser buena.
 
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

# Submissions

|            | Version |      | F1 Test |        | F1 Val  |        | Nº de hojas |        | Nº instancias | % Val       | CV % accuracy K=5 C4.5 |
|------------|---------|------|---------|--------|---------|--------|-------------|--------|---------------|-------------| ---------------------- |
| Submission | CART    | C4.5 | CART    | C4.5   | CART    | C4.5   | CART        | C4.5   |               | CART | C4.5 |
| 1          | 3       | 3    | 0.4959  | 0.3364 | 0.9066  | 0.9593 | 23          | 724    | 81.120        | 20   | 20   | 92.29                  |
| 2          | 4       | 4    | 0.5356  | 0.6581 | 0.8855  | 0.9553 | 42          | 1.387  | 79.920        | 20   | 20   | 90.81                  |
| 3          |         | 5    |         | 0.6778 | 0.8756  | 0.9425 | 25          | 571    | 74.973        | 20   | 20   | 89.18                  |
| 4          |         | 6    |         | 0.6836 | -       | 0.6814 | -           | 5.586  | 179.275       | -    |  5   | 64.85                  |
| 5          |         | 7    |         | 0.5064 |         | 0.8724 |             | 3.896  | 125.377       |      |  5   | 93.20                  |
| 6          |         | 8    |         | 0.5812 |         | 0.9328 |             | 10.641 | 302.703       |      | 35   | 72.17                  |
| 7          |         | 9    |         | 0.5703 |         | 0.8512 |             | 12.790 | 302.703       |      | 35   | 70.23                  |
| 8          |         | 10   |         | 0.6734 |         | 0.7018 |             | 4.821  | 108.611       |      |  5   | 62.25                  |
| 9          |         | 11   |         | 0.6285 |         | 0.6388 |             | 1.995  | 103.180       |      |  5   | 61.72                  |

| 10         |         | 6    |         | 0.6848 |         | 0.6746 |             | 2.040  | 179.275       |      |  5   | 65.43                  |
| 11         |         | 6    |         | 0.6850*|         | 0.6781 |             | 939    | 179.275       |      |  5   | 65.30                  |
| 12         |         | 6    |         | 0.6803 |         | 0.6687 |             | 462    | 179.275       |      |  5   | 64.90                  |


v6-3 C=0.15, M=20
f1 test 0.6850
M: 20[1] "Number of leaves: 939\nSize of the tree: 1392\n"
$f1 [1] 0.6781512
Correctly Classified Instances      111228               65.3088 %

v6-4 C=0.1 M=50
test 0.6803
[1] "Number of leaves: 462\nSize of the tree: 655\n"
f1 [1] 0.6687599
Correctly Classified Instances      110538               64.9036 %



-C <pruning confidence>
  Set confidence threshold for pruning.
  (default 0.25)
 
 -M <minimum number of instances>
  Set minimum number of instances per leaf.
  (default 2)

v6 - 2 -> C = 0.1
hojas 2040
f1 0.6746899
val 65.4332

C=0.05
val 65.4332
[1] "Number of leaves: 1223\nSize of the tree: 1884\n"
$f1
[1] 0.6699293

C=0.15
val 65.4332
[1] "Number of leaves: 2952\nSize of the tree: 4674\n"
$f1
[1] 0.6773581

C=0.20
val 65.4332
[1] "Number of leaves: 4200\nSize of the tree: 6671\n"
$f1
[1] 0.678912


Más bajo el C, menos se predice la clase 1
Más alto el M, más simple el árbol (más pequeño)

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

- https://stats.stackexchange.com/questions/28029/training-a-decision-tree-against-unbalanced-data


Imblearn Documentation
- https://github.com/scikit-learn-contrib/imbalanced-learn
Undersampling for Imbalanced Classification
- https://machinelearningmastery.com/undersampling-algorithms-for-imbalanced-classification/
Multi-class Imbalanced Classification
- https://machinelearningmastery.com/multi-class-imbalanced-classification/

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



C: 0.1
M: 5[1] "Number of leaves: 1383\nSize of the tree: 2135\n"
Accuracy global:
[1] 0.6501562
        predicted
expected    1    2    3
       1  391  648   15
       2  225 4021  750
       3   19 1479 1416
$precision
        1         2         3 
0.6347403 0.9952970 0.4891192 

$recall
        1         2         3 
0.3763234 0.9962834 0.6537396 

$f1
[1] 0.6759569

=== 3 Fold Cross Validation ===

=== Summary ===

Correctly Classified Instances      111201               65.2929 %
Incorrectly Classified Instances     59110               34.7071 %
Kappa statistic                          0.3383
Mean absolute error                      0.3046
Root mean squared error                  0.3953
Relative absolute error                 80.6125 %
Root relative squared error             90.9342 %
Total Number of Instances           170311     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.363    0.029    0.621      0.363    0.458      0.425    0.871     0.506     1
                 0.814    0.546    0.658      0.814    0.728      0.289    0.677     0.690     2
                 0.476    0.122    0.648      0.476    0.549      0.390    0.778     0.613     3
Weighted Avg.    0.653    0.350    0.650      0.653    0.639      0.337    0.732     0.644     

=== Confusion Matrix ===

     a     b     c   <-- classified as
  7232 12382   308 |     a = 1
  4069 78051 13785 |     b = 2
   338 28228 25918 |     c = 3
_______________________________________________________________________
C: 0.15
M: 5[1] "Number of leaves: 1802\nSize of the tree: 2826\n"
Accuracy global:
[1] 0.6532798
        predicted
expected    1    2    3
       1  410  631   13
       2  230 4016  750
       3   19 1465 1430
$precision
        1         2         3 
0.6406250 0.9952912 0.4939551 

$recall
        1         2         3 
0.3938521 0.9967734 0.6559633 

$f1
[1] 0.6824611

=== 3 Fold Cross Validation ===

=== Summary ===

Correctly Classified Instances      111241               65.3164 %
Incorrectly Classified Instances     59070               34.6836 %
Kappa statistic                          0.3416
Mean absolute error                      0.3034
Root mean squared error                  0.396 
Relative absolute error                 80.2857 %
Root relative squared error             91.0911 %
Total Number of Instances           170311     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.373    0.031    0.618      0.373    0.465      0.430    0.871     0.503     1
                 0.808    0.537    0.660      0.808    0.726      0.291    0.675     0.688     2
                 0.484    0.126    0.644      0.484    0.552      0.391    0.776     0.610     3
Weighted Avg.    0.653    0.346    0.650      0.653    0.640      0.339    0.730     0.641     

=== Confusion Matrix ===

     a     b     c   <-- classified as
  7428 12152   342 |     a = 1
  4225 77465 14215 |     b = 2
   365 27771 26348 |     c = 3
_______________________________________________________________________
C: 0.2
M: 5[1] "Number of leaves: 2620\nSize of the tree: 4074\n"
Accuracy global:
[1] 0.6506024
        predicted
expected    1    2    3
       1  413  627   14
       2  233 3984  779
       3   18 1461 1435
$precision
        1         2         3 
0.6393189 0.9955022 0.4955110 

$recall
        1         2         3 
0.3971154 0.9964982 0.6481481 

$f1
[1] 0.6825203

=== 3 Fold Cross Validation ===

=== Summary ===

Correctly Classified Instances      110930               65.1338 %
Incorrectly Classified Instances     59381               34.8662 %
Kappa statistic                          0.3373
Mean absolute error                      0.3026
Root mean squared error                  0.3969
Relative absolute error                 80.0763 %
Root relative squared error             91.2976 %
Total Number of Instances           170311     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.366    0.030    0.616      0.366    0.460      0.425    0.869     0.503     1
                 0.808    0.541    0.658      0.808    0.725      0.287    0.674     0.684     2
                 0.480    0.126    0.642      0.480    0.549      0.387    0.775     0.609     3
Weighted Avg.    0.651    0.348    0.648      0.651    0.638      0.335    0.729     0.639     

=== Confusion Matrix ===

     a     b     c   <-- classified as
  7299 12265   358 |     a = 1
  4186 77482 14237 |     b = 2
   358 27977 26149 |     c = 3
_______________________________________________________________________
C: 0.1
M: 10[1] "Number of leaves: 1013\nSize of the tree: 1539\n"
Accuracy global:
[1] 0.6477019
        predicted
expected    1    2    3
       1  381  657   16
       2  229 4018  749
       3   17 1490 1407
$precision
        1         2         3 
0.6245902 0.9957869 0.4856748 

$recall
        1         2         3 
0.3670520 0.9960337 0.6525974 

$f1
[1] 0.6717286

=== 3 Fold Cross Validation ===

=== Summary ===

Correctly Classified Instances      111034               65.1948 %
Incorrectly Classified Instances     59277               34.8052 %
Kappa statistic                          0.3407
Mean absolute error                      0.3063
Root mean squared error                  0.3946
Relative absolute error                 81.0474 %
Root relative squared error             90.785  %
Total Number of Instances           170311     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.373    0.031    0.612      0.373    0.464      0.427    0.872     0.507     1
                 0.803    0.533    0.660      0.803    0.725      0.288    0.678     0.695     2
                 0.488    0.129    0.640      0.488    0.554      0.390    0.778     0.616     3
Weighted Avg.    0.652    0.345    0.648      0.652    0.639      0.337    0.733     0.648     

=== Confusion Matrix ===

     a     b     c   <-- classified as
  7428 12157   337 |     a = 1
  4302 77013 14590 |     b = 2
   398 27493 26593 |     c = 3
_______________________________________________________________________
C: 0.15
M: 10[1] "Number of leaves: 1247\nSize of the tree: 1897\n"
Accuracy global:
[1] 0.6462517
        predicted
expected    1    2    3
       1  394  646   14
       2  236 4003  757
       3   18 1500 1396
$precision
        1         2         3 
0.6253968 0.9955235 0.4820442 

$recall
        1         2         3 
0.3788462 0.9965148 0.6483976 

$f1
[1] 0.6736187

=== 3 Fold Cross Validation ===

=== Summary ===

Correctly Classified Instances      111211               65.2988 %
Incorrectly Classified Instances     59100               34.7012 %
Kappa statistic                          0.342 
Mean absolute error                      0.3048
Root mean squared error                  0.3945
Relative absolute error                 80.64   %
Root relative squared error             90.7485 %
Total Number of Instances           170311     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.373    0.030    0.621      0.373    0.466      0.431    0.873     0.511     1
                 0.805    0.533    0.661      0.805    0.726      0.291    0.678     0.693     2
                 0.487    0.129    0.640      0.487    0.553      0.390    0.778     0.615     3
Weighted Avg.    0.653    0.345    0.650      0.653    0.640      0.339    0.733     0.647     

=== Confusion Matrix ===

     a     b     c   <-- classified as
  7428 12128   366 |     a = 1
  4123 77251 14531 |     b = 2
   401 27551 26532 |     c = 3
_______________________________________________________________________
C: 0.2
M: 10[1] "Number of leaves: 1577\nSize of the tree: 2426\n"
Accuracy global:
[1] 0.6481481
        predicted
expected    1    2    3
       1  394  646   14
       2  228 4008  760
       3   18 1488 1408
$precision
        1         2         3 
0.6334405 0.9955291 0.4861878 

$recall
        1         2         3 
0.3788462 0.9965191 0.6494465 

$f1
[1] 0.6754112

=== 3 Fold Cross Validation ===

=== Summary ===

Correctly Classified Instances      111155               65.2659 %
Incorrectly Classified Instances     59156               34.7341 %
Kappa statistic                          0.3413
Mean absolute error                      0.3034
Root mean squared error                  0.3945
Relative absolute error                 80.2847 %
Root relative squared error             90.7572 %
Total Number of Instances           170311     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.368    0.030    0.617      0.368    0.461      0.426    0.873     0.510     1
                 0.805    0.534    0.660      0.805    0.726      0.290    0.680     0.694     2
                 0.489    0.128    0.641      0.489    0.555      0.391    0.779     0.619     3
Weighted Avg.    0.653    0.345    0.649      0.653    0.640      0.338    0.734     0.649     

=== Confusion Matrix ===

     a     b     c   <-- classified as
  7329 12236   357 |     a = 1
  4173 77206 14526 |     b = 2
   379 27485 26620 |     c = 3
_______________________________________________________________________
C: 0.1
M: 20[1] "Number of leaves: 767\nSize of the tree: 1118\n"
Accuracy global:
[1] 0.6483713
        predicted
expected    1    2    3
       1  390  645   19
       2  232 4036  728
       3   20 1508 1386
$precision
        1         2         3 
0.6270096 0.9950690 0.4789219 

$recall
        1         2         3 
0.3768116 0.9953144 0.6556291 

$f1
[1] 0.6731454

=== 3 Fold Cross Validation ===

=== Summary ===

Correctly Classified Instances      111043               65.2001 %
Incorrectly Classified Instances     59268               34.7999 %
Kappa statistic                          0.3359
Mean absolute error                      0.3076
Root mean squared error                  0.3945
Relative absolute error                 81.3942 %
Root relative squared error             90.7434 %
Total Number of Instances           170311     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.365    0.030    0.616      0.365    0.459      0.424    0.872     0.507     1
                 0.816    0.550    0.657      0.816    0.728      0.288    0.678     0.695     2
                 0.469    0.119    0.649      0.469    0.544      0.387    0.777     0.617     3
Weighted Avg.    0.652    0.351    0.649      0.652    0.637      0.335    0.732     0.648     

=== Confusion Matrix ===

     a     b     c   <-- classified as
  7273 12324   325 |     a = 1
  4170 78239 13496 |     b = 2
   359 28594 25531 |     c = 3
_______________________________________________________________________
C: 0.15
M: 20[1] "Number of leaves: 939\nSize of the tree: 1392\n"
Accuracy global:
[1] 0.6481481
        predicted
expected    1    2    3
       1  402  637   15
       2  237 3981  778
       3   22 1465 1427
$precision
        1         2         3 
0.6291080 0.9945041 0.4934302 

$recall
        1         2         3 
0.3869105 0.9962462 0.6471655 

$f1
[1] 0.6781512

=== 3 Fold Cross Validation ===

=== Summary ===

Correctly Classified Instances      111228               65.3088 %
Incorrectly Classified Instances     59083               34.6912 %
Kappa statistic                          0.3407
Mean absolute error                      0.3066
Root mean squared error                  0.3939
Relative absolute error                 81.1285 %
Root relative squared error             90.6177 %
Total Number of Instances           170311     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.371    0.030    0.618      0.371    0.463      0.428    0.872     0.511     1
                 0.809    0.539    0.659      0.809    0.727      0.290    0.680     0.698     2
                 0.481    0.125    0.645      0.481    0.551      0.390    0.780     0.619     3
Weighted Avg.    0.653    0.347    0.650      0.653    0.640      0.338    0.734     0.651     

=== Confusion Matrix ===

     a     b     c   <-- classified as
  7383 12203   336 |     a = 1
  4184 77627 14094 |     b = 2
   370 27896 26218 |     c = 3
_______________________________________________________________________
C: 0.2
M: 20[1] "Number of leaves: 1083\nSize of the tree: 1615\n"
Accuracy global:
[1] 0.6491522
        predicted
expected    1    2    3
       1  399  641   14
       2  231 3994  771
       3   21 1467 1426
$precision
        1         2         3 
0.6333333 0.9947696 0.4929139 

$recall
        1         2         3 
0.3836538 0.9965070 0.6490669 

$f1
[1] 0.6779321

=== 3 Fold Cross Validation ===

=== Summary ===

Correctly Classified Instances      111075               65.2189 %
Incorrectly Classified Instances     59236               34.7811 %
Kappa statistic                          0.3405
Mean absolute error                      0.3052
Root mean squared error                  0.3938
Relative absolute error                 80.7567 %
Root relative squared error             90.6014 %
Total Number of Instances           170311     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.374    0.031    0.615      0.374    0.465      0.429    0.874     0.514     1
                 0.805    0.535    0.660      0.805    0.725      0.289    0.681     0.698     2
                 0.485    0.127    0.642      0.485    0.552      0.389    0.780     0.620     3
Weighted Avg.    0.652    0.346    0.649      0.652    0.639      0.337    0.735     0.651     

=== Confusion Matrix ===

     a     b     c   <-- classified as
  7454 12126   342 |     a = 1
  4293 77218 14394 |     b = 2
   374 27707 26403 |     c = 3
_______________________________________________________________________