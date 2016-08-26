/**
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Isak Karlsson
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
 * associated documentation files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package org.briljantframework.mimir.classification;

import static org.briljantframework.array.Arrays.doubleVector;
import static org.briljantframework.array.Arrays.mean;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.briljantframework.Check;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.Is;
import org.briljantframework.data.Na;
import org.briljantframework.data.series.DoubleSeries;
import org.briljantframework.data.series.Series;
import org.briljantframework.mimir.data.Output;
import org.briljantframework.mimir.data.Outputs;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ClassifierMeasure {

  private static final String PREDICTED_ACTUAL_SIZE =
      "Size of predicted and actual values does not match";
  private static final String ILLEGAL_SCORE_MATRIX = "Illegal score matrix";

  private final double accuracy;
  private final double areaUnderRocCurve;
  private final double brierScore;
  private final double precision;
  private final double recall;
  private final double fMeasure;

  /**
   * Compute classifier measures
   * 
   * @param predicted the predicated values
   * @param truth the true values
   */
  public ClassifierMeasure(Output<?> predicted, Output<?> truth) {
    this(predicted, truth, null, Outputs.unique(Arrays.asList(predicted, truth)));
  }

  /**
   * Compute some common classifier measures
   * 
   * @param predicted the predictions
   * @param truth the true values
   * @param scores an array of scores (one column per class; one row per instance)
   * @param classes a set of classes
   */
  public ClassifierMeasure(Output<?> predicted, Output<?> truth, DoubleArray scores,
      List<?> classes) {
    Check.argument(predicted.size() == truth.size(),
        "The predicted and actual values must have the same size.");
    Check.argument(classes != null, "If score matrix is given, classes are required");

    if (scores != null) {
      Check.argument(scores.rows() == predicted.size(), "Illegal score matrix (illegal rows)");
    }

    this.accuracy = accuracy(predicted, truth);
    this.precision = averagePrecision(predicted, truth, classes);
    this.recall = averageRecall(predicted, truth, classes);
    this.fMeasure = 2 * precision * recall / (precision + recall);

    if (scores != null) {
      areaUnderRocCurve = averageAreaUnderRocCurve(predicted, truth, scores, classes);
      brierScore = brierScore(predicted, truth, scores, classes);
    } else {
      brierScore = Na.DOUBLE;
      areaUnderRocCurve = Na.DOUBLE;
    }
  }

  private double averageRecall(Output<?> prediction, Output<?> truth, List<?> classes) {
    Check.argument(prediction.size() == truth.size(), "illegal size");
    double[] recall = new double[classes.size()];
    for (int i = 0; i < classes.size(); i++) {
      Object cls = classes.get(i);
      int predicted = 0, truePositive = 0;
      for (int j = 0; j < prediction.size(); j++) {
        if (Is.equal(truth.get(j), cls)) {
          predicted++;
          if (Is.equal(prediction.get(j), cls)) {
            truePositive++;
          }
        }
      }
      recall[i] = predicted > 0 ? (double) truePositive / predicted : 0;
    }
    return mean(doubleVector(recall));
  }

  public double averagePrecision(Output<?> prediction, Output<?> truth, List<?> classes) {
    Check.argument(prediction.size() == truth.size(), "illegal size");
    double[] precision = new double[classes.size()];
    for (int i = 0; i < classes.size(); i++) {
      int predicted = 0, truePositive = 0;
      Object cls = classes.get(i);
      for (int j = 0; j < prediction.size(); j++) {
        if (Is.equal(prediction.get(j), cls)) {
          predicted++;
          if (Is.equal(truth.get(j), cls)) {
            truePositive++;
          }
        }
      }
      precision[i] = predicted > 0 ? (double) truePositive / predicted : 0;
    }
    return mean(doubleVector(precision));
  }

  /**
   * Returns the prediction accuracy, i.e., the fraction of correctly classified examples. Object
   * 
   * @param p the predicted values; shape {@code [no sample]}
   * @param t the actual values; shape {@code [no samples]}
   * @return the accuracy
   */
  public static double accuracy(Output<?> p, Output<?> t) {
    Check.argument(p.size() == t.size(), PREDICTED_ACTUAL_SIZE);
    double accuracy = 0;

    int n = p.size();
    for (int i = 0; i < n; i++) {
      if (Is.equal(p.get(i), t.get(i))) {
        accuracy += 1;
      }
    }
    return accuracy / n;
  }

  /**
   * Computes the brier score. The brier score is defined as the squared difference between the
   * classification probabilities and the optimal probability.
   *
   * @param p vector of shape {@code [no samples]}
   * @param t vector of shape {@code [no samples]}
   * @param scores matrix of shape {@code [no samples, no classes]}
   * @param c vector of shape {@code [no classes]}; the i:th index gives the score column in
   *        {@code scores}
   * @return the brier score
   */
  public static double brierScore(Output<?> p, Output<?> t, DoubleArray scores, List<?> c) {
    Check.argument(scores.isMatrix() && scores.columns() == c.size() && scores.rows() == p.size(),
        ILLEGAL_SCORE_MATRIX);

    Check.argument(p.size() == t.size(), PREDICTED_ACTUAL_SIZE);

    int n = p.size();
    double brier = 0;
    for (int i = 0; i < n; i++) {
      // int classIndex = find(c, p, i);
      int classIndex = c.indexOf(p.get(i));
      if (classIndex < 0 || classIndex > c.size()) {
        throw new IllegalStateException("Missing class " + p.get(i));
      }

      double prob = scores.get(i, classIndex);
      if (Is.equal(p.get(i), t.get(i))) {
        brier += Math.pow(1 - prob, 2);
      } else {
        brier += prob * prob;
      }
    }
    return brier / n;
  }

  /**
   * Get the weighted average area under ROC curve.
   * 
   * @param p the predictions
   * @param a the actual values
   * @param score the probability scores
   * @param c the classes
   * @return the weighted area under ROC curve
   */
  public static double averageAreaUnderRocCurve(Output<?> p, Output<?> a, DoubleArray score,
      List<?> c) {
    Series auc = areaUnderRocCurve(p, a, score, c);
    Map<Object, Integer> dist = Outputs.valueCounts(a);
    double averageAuc = 0;
    for (Object classKey : auc.getIndex()) {
      if (dist.containsKey(classKey)) {
        double classCount = dist.get(classKey);
        averageAuc += auc.getDouble(classKey) * (classCount / a.size());
      }
    }
    return averageAuc;
  }

  /**
   * @param p vector of shape {@code [no samples]} the predicted labels
   * @param t vector of shape {@code [no samples]} the true labels
   * @param score matrix of shape {@code [no samples, domain.size()]} with scores (probabilities,
   *        confidences or binary indicators)
   * @param c vector of shape {@code [no classes]} the i:th index in the domain denotes the score in
   *        the j:th column of the score matrix
   * @return a vector of labels (from {@code c}) and its associated area under roc-curve
   */
  public static Series areaUnderRocCurve(Output<?> p, Output<?> t, DoubleArray score, List<?> c) {
    Check.argument(score.isMatrix() && score.columns() == c.size() && score.rows() == p.size(),
        ILLEGAL_SCORE_MATRIX);
    Check.argument(p.size() == t.size(), PREDICTED_ACTUAL_SIZE);
    Series.Builder builder = new DoubleSeries.Builder();
    for (int i = 0; i < c.size(); i++) {
      Object value = c.get(i);
      DoubleArray s = score.getColumn(i);
      builder.set(value, computeAuc(p, t, s, value));
    }
    return builder.build();
  }

  private static double computeAuc(Output<?> p, Output<?> t, DoubleArray score, Object label) {
    double truePositives = 0, falsePositives = 0, positives = 0;
    PredictionProbability[] pairs = new PredictionProbability[p.size()];
    for (int i = 0; i < t.size(); i++) {
      boolean positiveness = Is.equal(t.get(i), label);
      if (positiveness) {
        positives++;
      }
      pairs[i] = new PredictionProbability(positiveness, score.get(i));
    }

    // Sort in decreasing order of posterior probability
    Arrays.sort(pairs);

    double negatives = p.size() - positives;
    double previousProbability = -1;
    double auc = 0.0;
    double previousTruePositive = 0.0;
    double previousFalsePositive = 0.0;

    // Calculates the auc using trapezoidal rule
    for (PredictionProbability pair : pairs) {
      double probability = pair.probability;
      if (probability != previousProbability) {
        double falseChange = Math.abs(falsePositives - previousFalsePositive);
        double trueChange = truePositives + previousTruePositive;
        auc += falseChange * trueChange / 2;

        previousFalsePositive = falsePositives;
        previousTruePositive = truePositives;
        previousProbability = probability;
      }

      if (pair.positive) {
        truePositives++;
      } else {
        falsePositives++;
      }
    }
    if (truePositives == 0) {
      return 0;
    } else if (falsePositives == 0) {
      return 1;
    } else {
      double negChange = Math.abs(negatives - previousFalsePositive);
      double posChange = positives + previousTruePositive;
      return (auc + negChange * posChange / 2) / (positives * negatives);
    }
  }

  /**
   * Returns the prediction error, i.e. the fraction of miss-classified values. The same as
   * {@code 1 - accuracy}.
   *
   * @param p the predicted values; shape {@code [no sample]}
   * @param t the actual values; shape {@code [no samples]}
   * @return the error rate
   */
  public static double error(Output<Object> p, Output<Object> t) {
    return 1 - accuracy(p, t);
  }

  /**
   * @return the error
   */
  public double getError() {
    return 1 - getAccuracy();
  }

  /**
   * @return the accuracy
   */
  public double getAccuracy() {
    return accuracy;
  }

  /**
   * @return the average area under ROC curve
   */
  public double getAreaUnderRocCurve() {
    return areaUnderRocCurve;
  }

  /**
   * @return the mean square error of predicted probabilities
   */
  public double getBrierScore() {
    return brierScore;
  }

  /**
   * @return the macro averaged precision
   */
  public double getPrecision() {
    return precision;
  }

  /**
   * @return the macro averaged recall
   */
  public double getRecall() {
    return recall;
  }

  /**
   * @return the macro averaged f1 measure (harmonic mean between precision and recall)
   */
  public double getF1Measure() {
    return fMeasure;
  }

  @Override
  public String toString() {
    return "ClassifierMeasure{" + "accuracy=" + accuracy + ", areaUnderRocCurve="
        + areaUnderRocCurve + ", brierScore=" + brierScore + ", precision=" + precision
        + ", recall=" + recall + '}';
  }

  private static final class PredictionProbability implements Comparable<PredictionProbability> {

    public final boolean positive;
    public final double probability;

    private PredictionProbability(boolean positive, double probability) {
      this.positive = positive;
      this.probability = probability;
    }

    @Override
    public int compareTo(PredictionProbability o) {
      return Double.compare(o.probability, this.probability);
    }
  }
}
