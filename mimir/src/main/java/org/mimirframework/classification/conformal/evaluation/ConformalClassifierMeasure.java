package org.mimirframework.classification.conformal.evaluation;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.BooleanArray;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.data.vector.Vectors;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ConformalClassifierMeasure {

  private final double accuracy, error, averagePvalue, confidence, credibility, singletons,
      noClasses;

  public ConformalClassifierMeasure(Vector truth, DoubleArray score, double significance,
      Vector classes) {
    // Compute confidence and credibility
    double correct = 0;
    double avgConfidence = 0;
    double avgCredibility = 0;
    double noSingletons = 0;
    double avgPValue = 0;
    double avgNoClasses = 0;
    for (int i = 0; i < score.rows(); i++) {
      DoubleArray estimate = score.getRow(i);
      BooleanArray predictions = estimate.where(p -> p >= significance);
      correct += predictions.get(Vectors.find(classes, truth, i)) ? 1 : 0;
      int prediction = Arrays.argmax(estimate);
      double credibility = estimate.get(prediction);
      double confidence = 1 - Arrays.maxExcluding(estimate, prediction);

      avgCredibility += credibility / score.rows();
      avgConfidence += confidence / score.rows();
      avgPValue += Arrays.mean(estimate) / score.rows();

      double noPredictions = Arrays.sum(predictions);
      if (noPredictions == 1) {
        noSingletons++;
      }
      avgNoClasses += noPredictions / score.rows();
    }
    accuracy = correct / truth.size();
    error = 1 - accuracy;
    averagePvalue = avgPValue;
    confidence = avgConfidence;
    credibility = avgCredibility;
    singletons = noSingletons / truth.size();
    noClasses = avgNoClasses;
  }

  public double getAccuracy() {
    return accuracy;
  }

  public double getError() {
    return error;
  }

  public double getAveragePvalue() {
    return averagePvalue;
  }

  public double getConfidence() {
    return confidence;
  }

  public double getCredibility() {
    return credibility;
  }

  public double getSingletons() {
    return singletons;
  }

  public double getNoClasses() {
    return noClasses;
  }

  @Override
  public String toString() {
    return "ConformalClassifierMeasure{" + "accuracy=" + accuracy + ", error=" + error
        + ", averagePvalue=" + averagePvalue + ", confidence=" + confidence + ", credibility="
        + credibility + ", singletons=" + singletons + ", noClasses=" + noClasses + '}';
  }
}
