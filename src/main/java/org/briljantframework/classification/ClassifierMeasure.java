package org.briljantframework.classification;

import static org.briljantframework.data.vector.Vectors.find;

import java.util.Arrays;

import org.briljantframework.Check;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.Is;
import org.briljantframework.data.Na;
import org.briljantframework.data.vector.DoubleVector;
import org.briljantframework.data.vector.Vector;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ClassifierMeasure {

  private final double accuracy, areaUnderRocCurve, brierScore, precision, recall;

  public ClassifierMeasure(Vector t, Vector p, DoubleArray scores, Vector classes) {
    this.accuracy = accuracy(p, t);
    this.precision = 0;
    this.recall = 0;

    if (scores != null) {
      areaUnderRocCurve = averageAreaUnderRocCurve(p, t, scores, classes);
      brierScore = brierScore(p, t, scores, classes);
    } else {
      brierScore = Na.DOUBLE;
      areaUnderRocCurve = Na.DOUBLE;
    }
  }

  public ClassifierMeasure(Vector predictions, Vector truth) {
    this(truth, predictions, null, null);
  }

  public double getAccuracy() {
    return accuracy;
  }

  public double getError() {
    return 1 - getAccuracy();
  }

  public double getAreaUnderRocCurve() {
    return areaUnderRocCurve;
  }

  public double getBrierScore() {
    return brierScore;
  }

  public double getPrecision() {
    return precision;
  }

  public double getRecall() {
    return recall;
  }

  @Override
  public String toString() {
    return "ClassifierMeasure{" + "accuracy=" + accuracy + ", areaUnderRocCurve="
        + areaUnderRocCurve + ", brierScore=" + brierScore + ", precision=" + precision
        + ", recall=" + recall + '}';
  }

  /**
   * Returns the prediction error, i.e. the fraction of miss-classified values. The same as
   * {@code 1 - accuracy}.
   *
   * @param p the predicted values; shape {@code [no sample]}
   * @param t the actual values; shape {@code [no samples]}
   * @return the error rate
   */
  public static double error(Vector p, Vector t) {
    return 1 - accuracy(p, t);
  }

  /**
   * Returns the prediction accuracy, i.e., the fraction of correctly classified examples.
   *
   * @param p the predicted values; shape {@code [no sample]}
   * @param t the actual values; shape {@code [no samples]}
   * @return the accuracy
   */
  public static double accuracy(Vector p, Vector t) {
    Check.size(p.size(), t.size());
    double accuracy = 0;
    int n = p.size();
    for (int i = 0; i < n; i++) {
      if (Is.equal(p.loc().get(i), t.loc().get(i))) {
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
  public static double brierScore(Vector p, Vector t, DoubleArray scores, Vector c) {
    Check.size(p.size(), t.size());
    Check.size(t.size(), scores.rows());

    int n = p.size();
    double brier = 0;
    for (int i = 0; i < n; i++) {
      int classIndex = find(c, p, i);
      if (classIndex < 0 || classIndex > c.size()) {
        throw new IllegalStateException("Missing class " + p.loc().get(i));
      }

      double prob = scores.get(i, classIndex);
      if (Is.equal(p.loc().get(i), t.loc().get(i))) {
        brier += Math.pow(1 - prob, 2);
      } else {
        brier += prob * prob;
      }
    }
    return brier / n;
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
  public static Vector areaUnderRocCurve(Vector p, Vector t, DoubleArray score, Vector c) {
    Vector.Builder builder = new DoubleVector.Builder();
    for (int i = 0; i < c.size(); i++) {
      Object value = c.loc().get(i);
      DoubleArray s = score.getColumn(i);
      builder.set(value, computeAuc(p, t, s, value));
    }
    return builder.build();
  }

  public static double averageAreaUnderRocCurve(Vector p, Vector a, DoubleArray score, Vector c) {
    Vector auc = areaUnderRocCurve(p, a, score, c);
    Vector dist = a.valueCounts();
    double averageAuc = 0;
    for (Object classKey : auc.getIndex()) {
      if (dist.getIndex().contains(classKey)) {
        int classCount = dist.getAsInt(classKey);
        averageAuc += auc.getAsDouble(classKey) * (classCount / (double) a.size());
      }
    }
    return averageAuc;
  }

  private static double computeAuc(Vector p, Vector t, DoubleArray score, Object label) {
    double truePositives = 0, falsePositives = 0, positives = 0;
    PredictionProbability[] pairs = new PredictionProbability[p.size()];
    for (int i = 0; i < t.size(); i++) {
      boolean positiveness = Is.equal(t.loc().get(i), label);
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

  public String getName() {
    return null;
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
