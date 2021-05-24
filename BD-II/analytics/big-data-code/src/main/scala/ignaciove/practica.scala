// Ignacio Vellido Expósito
package main.scala.ignaciove

// ----------------------------------------------------------------------------
// Imports
// ----------------------------------------------------------------------------

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.{SparkConf, SparkContext}
import java.io.{File, PrintWriter}

// To get a timestamp
import java.util.Calendar
import java.text.SimpleDateFormat

// Parser
import main.scala.djgarcia.KeelParser

// Metrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// ----------------------------------------------------------------------------
// Main class
// ----------------------------------------------------------------------------

object practica extends Serializable {
  // --------------------------------------------------------------------------
  // Parameters
  val inCluster = true
  /**
    * 0 = No preprocessing
    * 1 = ROS
    * 2 = RUS
    */
  var preprocessing = 0
  var overRate = 0.5

  /**
    * 0 = No noise filtering
    * 1 = HME
    * 2 = NCNEdit
    */
  var noiseFiltering = 0

  /**
    * 0 = No instance selection
    * 1 = FCNN
    * 2 = SSMA
    */
  var instanceSelectionModel = 0

  /**
    * 0 = No feature selection
    * 1 = ChiSq
    * 2 = PCA
    */
  var featureSelectionModel = 0

  /**
    * 0 = Decision Tree
    * 1 = Random Forest
    * 2 = PCARD
    */
  var model = 0

  var finalInstances = 0.0
  // --------------------------------------------------------------------------

  /**
    * Launch script
    *
    * @param arg
    */
  def main(args: Array[String]) {
      // Parse arguments
      // Usage: <model> <noiseFiltering> <instanceSelection> <preprocessing> <featureSelection> <oversamplingRate>
      args.length match {
        case 6 => {
          model = args(0).toInt
          noiseFiltering = args(1).toInt
          instanceSelectionModel = args(2).toInt
          preprocessing = args(3).toInt
          featureSelectionModel = args(4).toInt
          overRate = args(5).toInt
        }
        case 5 => {
          model = args(0).toInt
          noiseFiltering = args(1).toInt
          instanceSelectionModel = args(2).toInt
          preprocessing = args(3).toInt
          featureSelectionModel = args(4).toInt
        }
        case 4 => {
          model = args(0).toInt
          noiseFiltering = args(1).toInt
          instanceSelectionModel = args(2).toInt
          preprocessing = args(3).toInt
        }
        case 3 => {
          model = args(0).toInt
          noiseFiltering = args(1).toInt
          instanceSelectionModel = args(2).toInt
        }
        case 2 => {
          model = args(0).toInt
          noiseFiltering = args(1).toInt
        }
        case 1 => {
          model = args(0).toInt
        }
        case _ =>
      }


      // Basic setup
      val jobName = "BD-II: Practica"

      // Spark Configuration
      val conf = new SparkConf().setAppName(jobName)
      val sc = new SparkContext(conf)

      // Log level
      sc.setLogLevel("ERROR")

      // Load data
      val trainAndTest = loadDataset(sc)
      println("Loaded dataset with " + trainAndTest._1.count() + " train instances")

      // Preprocess data
      val features = featureSelection(trainAndTest._1, trainAndTest._2)
      var train = features._1
      var test  = features._2

      if (preprocessing != 0) {
        train = preprocess(train)
      }
      if (noiseFiltering != 0) {
        train = noiseFilter(train)
      }
      if (instanceSelectionModel != 0) {
        train = instanceSelection(train)
      }

      finalInstances = train.count()
      println("Applied preprocessing with " + finalInstances + " train final instances")
      train.persist()

      // Train
      val labelsAndPreds = trainModel(train, test, sc)
      println("Model trained and " + labelsAndPreds.count + " predictions made")

      labelsAndPreds.persist()
      train.unpersist()

      // Evaluate and write results
      val results = evaluate(labelsAndPreds)

      println("Experiment results calculated")
      writeResults(results)
  }
  
  /**
    * Load train and test datasets
    */
  def loadDataset(sc: SparkContext): (RDD[LabeledPoint], RDD[LabeledPoint]) = {
    var header    = ""
    var pathTrain = ""
    var pathTest  = ""

    // Para el cluster
    if (inCluster) {
      header = "/user/datasets/master/susy/susy.header"
      pathTrain = "/user/datasets/master/susy/susyMaster-Train.data"
      pathTest = "/user/datasets/master/susy/susyMaster-Test.data"
    }
    // Para la máquina virtual
    else {
      header = "file:///home/administrador/datasets/susy.header"
      pathTrain = "file:///home/administrador/datasets/susy-10k-tra.data"
      pathTest = "file:///home/administrador/datasets/susy-10k-tst.data"
    }

    // Load train and test with KeelParser 
    val converter = new KeelParser(sc, header)
    val train = sc.textFile(pathTrain, 10).map(line => converter.parserToLabeledPoint(line)).persist
    val test  = sc.textFile(pathTest, 10).map(line => converter.parserToLabeledPoint(line)).persist

    (train, test)
  }

  /**
    * Train a model and get predictions
    *
    * @param train
    * @param test
    */
  def trainModel(train: RDD[LabeledPoint], test: RDD[LabeledPoint], sc: SparkContext): RDD[(Double, Double)] = {
    model match {
      case 0 => Models.decisionTree(train, test)
      case 1 => Models.randomForest(train, test)
      case 2 => Models.pcard(train, test, sc)
      case 3 => Models.knn(train, test, sc)
    }
  }

  def preprocess(train: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    preprocessing match {
      case 0 => train
      case 1 => Preprocessor.ROS(train, overRate)
      case 2 => Preprocessor.RUS(train)
    }
  }

  def noiseFilter(train: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    noiseFiltering match {
      case 0 => train
      case 1 => Preprocessor.HME(train)
      case 2 => Preprocessor.NCNEdit(train)
    }
  }

  def instanceSelection(train: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    instanceSelectionModel match {
      case 0 => train
      case 1 => Preprocessor.FCNN(train)
      case 2 => Preprocessor.SSMA(train)
    }
  }

  def featureSelection(train: RDD[LabeledPoint], test: RDD[LabeledPoint]): (RDD[LabeledPoint], RDD[LabeledPoint]) = {
    featureSelectionModel match {
      case 0 => (train, test)
      case 1 => Preprocessor.ChiSq(train, test)
      case 2 => Preprocessor.PCA(train, test)
    }
  }

  /**
    * Calculates TPR x TNR
    */
  def evaluate(predsAndLabels: RDD[(Double, Double)]): String = {
    val metrics = new MulticlassMetrics(predsAndLabels)
    val cm = metrics.confusionMatrix
    
    val tpr = cm(0,0) / (cm(0,0) + cm(0,1))
    val tnr = cm(1,1) / (cm(1,1) + cm(1,0))

    "TPR x TNR:\t" + tpr * tnr + "\n" +
      "Accuracy:\t" + metrics.accuracy + "\n" +
      "Weighted FMeasure:\t" + metrics.weightedFMeasure + "\n" +
      "Confusion Matrix:\n" + cm.toString() + "\n"
  }

  /**
    * Write results in disk
    */
  def writeResults(results: String) = {
    // Get path
    var path = ""
    if (inCluster) {
      path = "/home/x79056166/spark/results/results-"
    }
    else {
      path = "/home/administrador/Descargas/practica/results/results-"
    }

    // Get date
    val format = new SimpleDateFormat("dd-hh:mm")
    val date = format.format(Calendar.getInstance().getTime())

    // Get experiment details
    val m = model match {
      case 0 => "decisionTree"
      case 1 => "randomForest"
      case 2 => "pcard"
      case 3 => "knn"
    }
    
    val f = featureSelectionModel match {
      case 0 => "None"
      case 1 => "ChiSq"
      case 2 => "PCA"
    }

    val p = preprocessing match {
      case 0 => "None"
      case 1 => "ROS"
      case 2 => "RUS"
    }

    val n = noiseFiltering match {
      case 0 => "None"
      case 1 => "HME"
      case 2 => "NCNEdit"
    }

    val i = instanceSelectionModel match {
      case 0 => "None"
      case 1 => "FCNN"
      case 2 => "SSMA"
    }

    // Append results
    val experimentInfo = "Date:\t" + date + "\n" + 
      "------------------------\n" +
      "Feature selection:\t\t" + f + "\n" +
      "Preprocessing:\t" + p + "\n" +
      "Oversampling Rate (only when preprocessing=1): " + overRate + "\n" +
      "Noise filtering:\t" + n + "\n" +
      "Instance selection:\t" + i + "\n" +
      "------------------------\n" +
      "Learning model:\t" + m + "\n" +
      "Number of final train instances: " + finalInstances + "\n" +
      "------------------------\n"

    val writer = new PrintWriter(path + date + ".txt")
    writer.write(experimentInfo + results + "------------------------------------------\n")
    writer.close()
  }
}