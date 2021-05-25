package main.scala.ignaciove

// ----------------------------------------------------------------------------
// Imports
// ----------------------------------------------------------------------------

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{DecisionTree, RandomForest}

// PCARD
import org.apache.spark.mllib.tree.PCARD
import org.apache.spark.SparkContext


import org.apache.spark.mllib.classification.kNN_IS.kNN_IS

// ----------------------------------------------------------------------------
// Main class
// ----------------------------------------------------------------------------

/**
 * Apply different learning algorithms to a dataset
 *
 * @author Ignacio Vellido
 */
object Models extends Serializable {
  /**
   * Train a decision tree model and returns predictions over a test set
   *
   * @param train
   * @param test
   */
  def decisionTree(train: RDD[LabeledPoint], test: RDD[LabeledPoint]): RDD[(Double, Double)] = {
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    var categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 8
    val maxBins = 32

    val modelDT = DecisionTree.trainClassifier(train, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val predsAndLabelsDT = test.map { point =>
      val prediction = modelDT.predict(point.features)
      (point.label, prediction)
    }.map { case (v, k) => (k, v) }

    predsAndLabelsDT
  }

  /**
   * Train a random forest model and returns predictions over a test set
   *
   * @param train
   * @param test
   */
  def randomForest(train: RDD[LabeledPoint], test: RDD[LabeledPoint]): RDD[(Double, Double)] = {
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    var categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 150
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 8
    val maxBins = 32

    val modelRF = RandomForest.trainClassifier(train, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Evaluate model on test instances
    val predsAndLabelsRF = test.map { point =>
      val prediction = modelRF.predict(point.features)
      (point.label, prediction)
    }.map { case (v, k) => (k, v) }

    predsAndLabelsRF
  }

  /**
   * Train a pcard model and returns predictions over a test set
   *
   * @param train
   * @param test
   */
  def pcard(train: RDD[LabeledPoint], test: RDD[LabeledPoint], sc: SparkContext): RDD[(Double, Double)] = {
    val cuts = 5
    val trees = 15

    val pcardTrain = PCARD.train(train, trees, cuts)
    
    val predictions = pcardTrain.predict(test)


    val predsAndLabelsPCARD = sc.parallelize(predictions)
                            .zipWithIndex
                            .map { case (v, k) => (k, v) }
                            .join(
                              test
                              .zipWithIndex
                              .map { case (v, k) => (k, v.label) }
                            )
                            .map(_._2)
    
    predsAndLabelsPCARD
  }

  /**
   * Train a knn model and returns predictions over a test set
   *
   * @param train
   * @param test
   */
  def knn(train: RDD[LabeledPoint], test: RDD[LabeledPoint], sc: SparkContext): RDD[(Double, Double)] = {
    // Parameters
    val k = 7
    val dist = 2 //euclidean
    val numClass = 2
    val numFeatures = train.first.features.size
    val numPartitionMap = 10
    val numReduces = 2
    val numIterations = 1
    val maxWeight = 5

    // Initialize the classifier
    val knn = kNN_IS.setup(train, test, k, dist, numClass, numFeatures, numPartitionMap, numReduces, numIterations, maxWeight)

    // Classify
    knn.predict(sc)
  }
}