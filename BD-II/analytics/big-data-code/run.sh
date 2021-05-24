#!/bin/bash

# Usage: ./run.sh <model> <noiseFiltering> <instanceSelection> <preprocessing> <featureSelection> <oversamplingRate>

# 0 = No preprocessing
# 1 = ROS
# 2 = RUS
 
# 0 = No noise filtering
# 1 = HME
# 2 = NCNEdit
 
# 0 = No instance selection
# 1 = FCNN
# 2 = SSMA
 
# 0 = Decision Tree
# 1 = Random Forest
# 2 = PCARD
# 3 = KNN
 
# 0 = No feature selection
# 1 = ChiSq
# 2 = PCA


# Local
/opt/spark-2.4.0/bin/spark-submit --master local[*] --class main.scala.ignaciove.practica ./target/Practica-1.0-jar-with-dependencies.jar $1 $2 $3 $4 $5 $6


# Cluster (.jar must be in cluster)
# /opt/spark-2.4.7/bin/spark-submit --master spark://hadoop-master:7077 --total-executor-cores 14 --executor-memory 46g --class main.scala.ignaciove.practica Practica-1.0-jar-with-dependencies.jar $1 $2 $3 $4 $5 $6 > log.txt 2>&1