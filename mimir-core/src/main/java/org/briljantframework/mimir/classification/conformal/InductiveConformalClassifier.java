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

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Set;

import org.briljantframework.Check;
import org.briljantframework.mimir.classification.ClassifierCharacteristic;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Output;
import org.briljantframework.mimir.supervised.AbstractLearner;
import org.briljantframework.mimir.supervised.Characteristic;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class InductiveConformalClassifier<In, Out> extends AbstractConformalClassifier<In, Out> {

  private final ClassifierNonconformity<In, Out> nonconformity;
  private ClassifierCalibrator<In,Out> calibrator;
  private ClassifierCalibratorScores<In> calibration = null;

  protected InductiveConformalClassifier(ClassifierNonconformity<In, Out> nonconformity,
      ClassifierCalibrator<In,Out> calibrator, boolean stochasticSmoothing, List<Out> classes) {
    super(stochasticSmoothing, classes);
    this.calibrator = Objects.requireNonNull(calibrator, "Calibrator is required.");
    this.nonconformity = Objects.requireNonNull(nonconformity, "Requires nonconformity scorer");
  }

  /**
   * Calibrate this inductive conformal classifier using the supplied data frame and output target
   *
   * @param x the data
   * @param y the calibration target
   */
  public void calibrate(Input<? extends In> x, Output<? extends Out> y) {
    calibration = calibrator.calibrate(nonconformity, x, y);
  }

  @Override
  public ClassifierNonconformity<In, Out> getClassifierNonconformity() {
    return nonconformity;
  }

  @Override
  protected ClassifierCalibratorScores<In> getCalibrationScores() {
    Check.state(calibration != null, "Classifier is not calibrated.");
    return calibration;
  }

  @Override
  public Set<Characteristic> getCharacteristics() {
    return Collections.singleton(ClassifierCharacteristic.ESTIMATOR);
  }

  /**
   * @author Isak Karlsson <isak-kar@dsv.su.se>
   */
  public static class Learner<In, Out>
      extends AbstractLearner<In, Out, InductiveConformalClassifier<In, Out>> {

    private final ClassifierNonconformity.Learner<In, Out, ? extends ClassifierNonconformity<In, Out>> learner;
    private final ClassifierCalibrator<In, Out> calibratable;

    public Learner(
        ClassifierNonconformity.Learner<In, Out, ? extends ClassifierNonconformity<In, Out>> learner,
        ClassifierCalibrator<In,Out> calibrator, boolean stochasticSmoothing) {
      set(STOCHASTIC_SMOOTHING, stochasticSmoothing);
      this.calibratable = Objects.requireNonNull(calibrator, "Calibrator is required.");
      this.learner = Objects.requireNonNull(learner, "Nonconformity learner is required.");
    }

    public Learner(
        ClassifierNonconformity.Learner<In, Out, ? extends ClassifierNonconformity<In, Out>> learner,
        ClassifierCalibrator<In,Out> calibrator) {
      this(learner, calibrator, true);
    }

    public Learner(
        ClassifierNonconformity.Learner<In, Out, ? extends ClassifierNonconformity<In, Out>> learner) {
      this(learner, ClassifierCalibrator.unconditional());
    }

    @Override
    public InductiveConformalClassifier<In, Out> fit(Input<? extends In> in,
        Output<? extends Out> out) {
      Objects.requireNonNull(in, "Input data is required.");
      Objects.requireNonNull(out, "Input target is required.");
      Check.argument(in.size() == out.size(),
          "The size of input data and input target don't match.");
      ClassifierNonconformity<In, Out> nc = learner.fit(in, out);
      return new InductiveConformalClassifier<>(nc, calibratable,
          getOrDefault(STOCHASTIC_SMOOTHING), nc.getClasses());
    }
  }
}
