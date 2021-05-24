package main.scala.ignaciove

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature._

/**
 * Apply different preprocessing methods to a RDD
 *
 * @author Ignacio Vellido
 */
object Preprocessor extends Serializable {
    /**
      * Apply Random Oversampling
      *
      * @param train
      * @param overRate
      */
    def ROS(train: RDD[LabeledPoint], overRate: Double): RDD[LabeledPoint] = {
        var oversample: RDD[LabeledPoint] = train.sparkContext.emptyRDD

        val train_positive = train.filter(_.label == 1)
        val train_negative = train.filter(_.label == 0)
        val num_neg = train_negative.count().toDouble
        val num_pos = train_positive.count().toDouble

        if (num_pos > num_neg) {
            val fraction = (num_pos * overRate) / num_neg
            oversample = train_positive.union(train_negative.sample(withReplacement = true, fraction))
        } else {
            val fraction = (num_neg * overRate) / num_pos
            oversample = train_negative.union(train_positive.sample(withReplacement = true, fraction))
        }
        oversample.repartition(train.getNumPartitions)
    }

    /**
      * Apply Random Undersampling
      *
      * @param train
      */
    def RUS(train: RDD[LabeledPoint]): RDD[LabeledPoint] = {
        var undersample: RDD[LabeledPoint] = train.sparkContext.emptyRDD

        val train_positive = train.filter(_.label == 1)
        val train_negative = train.filter(_.label == 0)
        val num_neg = train_negative.count().toDouble
        val num_pos = train_positive.count().toDouble

        if (num_pos > num_neg) {
            val fraction = num_neg / num_pos
            undersample = train_negative.union(train_positive.sample(withReplacement = false, fraction))
        } else {
            val fraction = num_pos / num_neg
            undersample = train_positive.union(train_negative.sample(withReplacement = false, fraction))
        }
        undersample
    }

    /**
      * Homogeneous ensemble
      *
      * @param train
      */
    def HME(train: RDD[LabeledPoint]): RDD[LabeledPoint] = {
        val nTrees = 100
        val maxDepthRF = 10
        val partitions = 4

        val hme_model = new HME_BD(train, nTrees, partitions, maxDepthRF, 48151623)

        hme_model.runFilter()
    }

    /**
      * NCNEdit Noise Filter
      *
      * @param train
      */
    def NCNEdit(train: RDD[LabeledPoint]): RDD[LabeledPoint] = {
        val k = 3 //number of neighbors

        val ncnedit_model = new NCNEdit_BD(train, k)

        ncnedit_model.runFilter()
    }
    
    /**
      * FCNN instance selection
      *
      * @param train
      */
    def FCNN(train: RDD[LabeledPoint]): RDD[LabeledPoint] = {
        val k = 3 //number of neighbors
        val fcnn_mr_model = new FCNN_MR(train, k)

        fcnn_mr_model.runPR()
    }

    /**
      * SSMA-SFLSDE instance selection
      *
      * @param train
      */
    def SSMA(train: RDD[LabeledPoint]): RDD[LabeledPoint] = {
        val ssmasflsde_mr_model = new SSMASFLSDE_MR(train)

        ssmasflsde_mr_model.runPR()
    }

    /**
      * Chi-square feature selection
      *
      * @param train
      */
    def ChiSq(train: RDD[LabeledPoint], test: RDD[LabeledPoint]): (RDD[LabeledPoint], RDD[LabeledPoint]) = {
        // Discritizing first
        val nBins = 25 // Number of bins

        val discretizerModel = new EqualWidthDiscretizer(train,nBins).calcThresholds()

        val discretizedTrain = discretizerModel.discretize(train)
        val discretizedTest = discretizerModel.discretize(test)

        // Chi-sq
        val numFeatures = 9
        val selector = new ChiSqSelector(numFeatures)
        val transformer = selector.fit(discretizedTrain)

        val chisqTrain = discretizedTrain.map { lp => 
          LabeledPoint(lp.label, transformer.transform(lp.features)) 
        }

        val chisqTest = discretizedTest.map { lp => 
          LabeledPoint(lp.label, transformer.transform(lp.features)) 
        }

        (chisqTrain, chisqTest)
    }

    /**
      * PCA feature selection
      *
      * @param train
      */
    def PCA(train: RDD[LabeledPoint], test: RDD[LabeledPoint]): (RDD[LabeledPoint], RDD[LabeledPoint]) = {
        val numFeatures = 9

        val pca = new PCA(9).fit(train.map(_.features))

        val projectedTrain = train.map(p => p.copy(features = pca.transform(p.features)))
        val projectedTest = test.map(p => p.copy(features = pca.transform(p.features)))

        (projectedTrain, projectedTest)
    }
}