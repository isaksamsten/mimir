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
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.util.Precision;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.classification.AbstractClassifier;

/**
 * Implements the basic p-value computation for (inductive) conformal predictors given the
 * 
 * @author Isak Karlsson
 */
public abstract class AbstractConformalClassifier<In> extends AbstractClassifier<In>
    implements ConformalClassifier<In> {

  private final boolean stochasticSmoothing;

  /**
   * Create a new conformal classifier.
   * 
   * @param stochasticSmoothing enable stochastic smoothing
   * @param classes the classes
   */
  protected AbstractConformalClassifier(boolean stochasticSmoothing, List<?> classes) {
    super(classes);
    this.stochasticSmoothing = stochasticSmoothing;
  }

  /**
   * Creates a new conformal classifier with stochastic smoothing enabled
   * 
   * @param classes the classes
   */
  protected AbstractConformalClassifier(List<?> classes) {
    this(true, classes);
  }

  /**
   * Return the non conformity scorer
   * 
   * @return a classifier nonconformity
   */
  protected abstract ClassifierNonconformity<In, Object> getClassifierNonconformity();

  /**
   * Get the calibration
   * 
   * @return a classifier calibration
   */
  protected abstract ClassifierCalibratorScores<In> getCalibrationScores();

  @Override
  public DoubleArray estimate(In example) {
    DoubleArray significance = DoubleArray.zeros(getClasses().size());
    double tau = stochasticSmoothing ? ThreadLocalRandom.current().nextDouble() : 1;
    for (int i = 0; i < significance.size(); i++) {
      Object label = getClasses().get(i);
      DoubleArray calibration = getCalibrationScores().get(example, label);
      double n = calibration.size() + 1;
      double nc = getClassifierNonconformity().estimate(example, label);
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

}
