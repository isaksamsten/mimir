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
import java.util.Objects;
import java.util.Set;

import org.briljantframework.Check;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.mimir.classification.ClassifierCharacteristic;
import org.briljantframework.mimir.supervised.Characteristic;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class InductiveConformalClassifier extends AbstractConformalClassifier {

  private final ClassifierNonconformity nonconformity;
  private ClassifierCalibrator calibrator;
  private ClassifierCalibratorScores calibration = null;

  protected InductiveConformalClassifier(ClassifierNonconformity nonconformity,
      ClassifierCalibrator calibrator, boolean stochasticSmoothing, Vector classes) {
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
  public void calibrate(DataFrame x, Vector y) {
    calibration = calibrator.calibrate(nonconformity, x, y);
  }

  @Override
  public ClassifierNonconformity getClassifierNonconformity() {
    return nonconformity;
  }

  @Override
  protected ClassifierCalibratorScores getCalibrationScores() {
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
  public static class Learner implements Predictor.Learner<InductiveConformalClassifier> {

    private final ClassifierNonconformity.Learner learner;
    private final boolean stochasticSmoothing;
    private final ClassifierCalibrator calibratable;

    public Learner(ClassifierNonconformity.Learner<? extends ClassifierNonconformity> learner,
        ClassifierCalibrator calibrator, boolean stochasticSmoothing) {
      this.stochasticSmoothing = stochasticSmoothing;
      this.calibratable = Objects.requireNonNull(calibrator, "Calibrator is required.");
      this.learner = Objects.requireNonNull(learner, "Nonconformity learner is required.");
    }

    public Learner(ClassifierNonconformity.Learner<? extends ClassifierNonconformity> learner,
        ClassifierCalibrator calibrator) {
      this(learner, calibrator, true);
    }

    public Learner(ClassifierNonconformity.Learner<? extends ClassifierNonconformity> learner) {
      this(learner, ClassifierCalibrator.unconditional());
    }

    @Override
    public InductiveConformalClassifier fit(DataFrame x, Vector y) {
      Objects.requireNonNull(x, "Input data is required.");
      Objects.requireNonNull(y, "Input target is required.");
      Check.argument(x.rows() == y.size(), "The size of input data and input target don't match.");
      ClassifierNonconformity nc = learner.fit(x, y);
      return new InductiveConformalClassifier(nc, calibratable, stochasticSmoothing,
          nc.getClasses());
    }

  }
}
