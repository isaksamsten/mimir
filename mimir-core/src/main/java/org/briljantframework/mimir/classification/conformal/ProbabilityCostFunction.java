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
package org.briljantframework.mimir.classification.conformal;

import java.util.List;

import org.briljantframework.Check;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.data.Output;

/**
 * A classification error function
 *
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
@FunctionalInterface
public interface ProbabilityCostFunction {

  static ProbabilityCostFunction margin() {
    return (score, y) -> {
      if (y < 0) {
        return 1;
      } else {
        return 0.5 - (score.get(y) - Arrays.maxExcluding(score, y)) / 2;
      }
    };
  }

  static ProbabilityCostFunction inverseProbability() {
    return (score, y) -> y < 0 ? 1 : 1 - score.get(y);
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
  static DoubleArray estimate(ProbabilityCostFunction pcf, DoubleArray scores, Output<?> y,
      List<?> classes) {
    Check.argument(classes.size() == scores.columns(), "Illegal prediction matrix");
    DoubleArray probabilities = DoubleArray.zeros(y.size());
    for (int i = 0, size = y.size(); i < size; i++) {
      int yIndex = classes.indexOf(y.get(i));// Vectors.find(classes, y, i);
      // if (yIndex < 0) {
      // Object label = y.loc().get(i);
      // throw new IllegalArgumentException(String.format("Illegal class: '%s' (not found)",
      // label));
      // }
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
