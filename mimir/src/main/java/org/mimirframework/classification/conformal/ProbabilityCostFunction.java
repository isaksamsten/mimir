package org.mimirframework.classification.conformal;

import java.util.Objects;

import org.briljantframework.Check;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.data.vector.Vectors;

/**
 * A classification error function
 *
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
@FunctionalInterface
public interface ProbabilityCostFunction {

  static ProbabilityCostFunction margin() {
    return (score, y) -> {
      Objects.requireNonNull(score, "Require predictions.");
      return 0.5 - (score.get(y) - Arrays.maxExcluding(score, y)) / 2;
    };
  }

  static ProbabilityCostFunction inverseProbability() {
    return (score, y) -> 1 - Objects.requireNonNull(score, "Require predictions").get(y);
  }

  /**
   * Compute the cost function for each row in the supplied score matrix
   * {@code [n-examples, n-classes]} and return a {@code [n-example]} array of costs.
   *
   * @param pcf
   * @param scores the score matrix (e.g., probability estimates)
   * @param y the true class array
   * @param classes the possible classes ({@code classes.loc().indexOf(y.loc().get(i))} is used to
   *        find the true class column in the score matrix for the i:th example)
   * @return an array of costs
   */
  static DoubleArray estimate(ProbabilityCostFunction pcf, DoubleArray scores, Vector y,
      Vector classes) {
    Check.argument(classes.size() == scores.columns(), "Illegal prediction matrix");
    DoubleArray probabilities = DoubleArray.zeros(y.size());
    for (int i = 0, size = y.size(); i < size; i++) {
      int yIndex = Vectors.find(classes, y, i);
      if (yIndex < 0) {
        Object label = y.loc().get(i);
        throw new IllegalArgumentException(String.format("Illegal class: '%s' (not found)", label));
      }
      double value = pcf.apply(scores.getRow(i), yIndex);
      probabilities.set(i, value);
    }

    return probabilities;
  }

  /**
   * Compute the cost function for the score array of shape {@code [n-classes]} given the specified
   * true class.
   * 
   * @param score the score array
   * @param y the true class index (i.e. the index in the score array which is considered the true
   *        class label)
   * @return the cost
   */
  double apply(DoubleArray score, int y);
}
