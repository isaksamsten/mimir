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

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

import org.apache.commons.math3.util.Precision;
import org.briljantframework.array.Array;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.Property;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.supervised.Parameterized;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public abstract class ConformalClassifier<In, Out> {

  public static final Property<Boolean> STOCHASTIC_SMOOTHING =
      Property.of("stochastic_smoothing", Boolean.class, true);

  public static final double SIGNIFICANCE = 0.05;
  private final boolean stochasticSmoothing;
  private final Array<Out> classes;

  /**
   * Create a new conformal classifier.
   *
   * @param classes the classes
   * @param stochasticSmoothing enable stochastic smoothing
   */
  protected ConformalClassifier(Array<Out> classes, boolean stochasticSmoothing) {
    this.classes = classes;
    this.stochasticSmoothing = stochasticSmoothing;
  }

  /**
   * Creates a new conformal classifier with stochastic smoothing enabled
   *
   * @param classes the classes
   */
  protected ConformalClassifier(Array<Out> classes) {
    this(classes, true);
  }


  public Array<Out> getClasses() {
    return classes;
  }

  /**
   * Returns the prediction of the given example or {@code NA}.
   *
   * @param record to which the class label shall be assigned
   * @return a class-label or {@code NA}
   */
  public Set<Out> predict(In record) {
    return predict(record, SIGNIFICANCE);
  }


  public List<Set<Out>> predict(Input<? extends In> x) {
    return predict(x, SIGNIFICANCE);
  }

  /**
   * Return the non conformity scorer
   *
   * @return a classifier nonconformity
   */
  protected abstract Nonconformity<In, Out> getClassifierNonconformity();

  /**
   * Get the calibration
   *
   * @return a classifier calibration
   */
  protected abstract CalibratorScores<In, Out> getCalibrationScores();

  /**
   * Estimates the the p-values associated with each class.
   *
   * @param in the vector to estimate the posterior probability for
   * @return the p-values
   */
  public DoubleArray estimate(In in) {
    DoubleArray significance = DoubleArray.zeros(getClasses().size());
    double tau = stochasticSmoothing ? ThreadLocalRandom.current().nextDouble() : 1;
    for (int i = 0; i < significance.size(); i++) {
      Out label = getClasses().get(i);
      DoubleArray calibration = getCalibrationScores().get(in, label);
      double n = calibration.size() + 1;
      double nc = getClassifierNonconformity().estimate(in, label);
      double gt = 0;
      double eq = 1;
      for (int j = 0; j < calibration.size(); j++) {
        double v = calibration.get(j);
        if (v > nc) {
          gt++;
        } else if (Precision.equals(v, nc, 10e-6)) {
          eq++;
        }
      }
      significance.set(i, (gt + eq * tau) / n);
    }
    return significance;
  }

  /**
   * Returns the conformal predictions for the records in the given data frame using the given
   * significance level.
   *
   * @param x to determine class labels for
   * @return a vector of class-labels for those records with which a label can be assigned with the
   *         given probability; or {@code NA}.
   */
  public List<Set<Out>> predict(Input<? extends In> x, double significance) {
    List<Set<Out>> predictions = new ArrayList<>();
    for (In in : x) {
      predictions.add(predict(in, significance));
    }
    return predictions;
  }

  /**
   * Returns the prediction of the given example or {@code NA}. A prediction is given iff one class
   * have a significance greater than or equal to the specified significance level.
   *
   * @param record to which the class label shall be assigned
   * @return a class-label or {@code NA}
   */
  public Set<Out> predict(In record, double significance) {
    DoubleArray estimate = estimate(record);
    Set<Out> set = new HashSet<>();
    for (int i = 0; i < estimate.size(); i++) {
      if (estimate.get(i) > significance) {
        set.add(getClasses().get(i));
      }
    }
    return set;
  }

  /**
   * Returns an {@code [n-samples, n-classes]} double array of p-values associated with each class.
   *
   * @param x the data frame of records to estimate the p-values
   * @return the p-values
   */
  public DoubleArray estimate(Input<? extends In> x) {
    DoubleArray estimations = DoubleArray.zeros(x.size(), getClasses().size());
    IntStream.range(0, x.size()).parallel().forEach(i -> estimations.setRow(i, estimate(x.get(i))));
    return estimations;
  }

  public static abstract class Learner<In, Out, P extends ConformalClassifier<In, Out>>
      extends Parameterized {

    public abstract P fit(Input<In> in, List<Out> out);
  }

}
