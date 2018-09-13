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
import java.util.Objects;

import org.briljantframework.Check;
import org.briljantframework.array.Array;
import org.briljantframework.mimir.classification.ProbabilityEstimator;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class InductiveConformalClassifier<In, Out> extends ConformalClassifier<In, Out> {

  private final Nonconformity<In, Out> nonconformity;
  private Calibrator<In, Out> calibrator;
  private CalibratorScores<In, Out> calibration = null;

  private InductiveConformalClassifier(Array<Out> classes, Nonconformity<In, Out> nonconformity,
      Calibrator<In, Out> calibrator, boolean stochasticSmoothing) {
    super(classes, stochasticSmoothing);
    this.calibrator = Objects.requireNonNull(calibrator, "Calibrator is required.");
    this.nonconformity = Objects.requireNonNull(nonconformity, "Requires nonconformity scorer");
  }

  /**
   * Calibrate this inductive conformal classifier using the supplied input and output.
   *
   * @param x the input
   * @param y the calibration target
   */
  public void calibrate(Input<In> x, List<Out> y) {
    calibration = calibrator.calibrate(nonconformity, x, y);
  }

  @Override
  public Nonconformity<In, Out> getClassifierNonconformity() {
    return nonconformity;
  }

  @Override
  protected CalibratorScores<In, Out> getCalibrationScores() {
    Check.state(calibration != null, "Classifier is not calibrated.");
    return calibration;
  }

  /**
   * @author Isak Karlsson <isak-kar@dsv.su.se>
   */
  public static class Learner<In, Out>
      extends ConformalClassifier.Learner<In, Out, InductiveConformalClassifier<In, Out>> {

    private final Nonconformity.Learner<In, Out> learner;
    private final Calibrator<In, Out> calibrator;

    public Learner(Nonconformity.Learner<In, Out> learner, Calibrator<In, Out> calibrator,
        boolean stochasticSmoothing) {
      set(STOCHASTIC_SMOOTHING, stochasticSmoothing);
      this.calibrator = Objects.requireNonNull(calibrator, "Calibrator is required.");
      this.learner = Objects.requireNonNull(learner, "Nonconformity learner is required.");
    }

    public Learner(Nonconformity.Learner<In, Out> learner, Calibrator<In, Out> calibrator) {
      this(learner, calibrator, true);
    }

    public Learner(Nonconformity.Learner<In, Out> learner) {
      this(learner, Calibrator.unconditional());
    }

    public Learner(Predictor.Learner<In, Out, ? extends ProbabilityEstimator<In, Out>> pet) {
      this(new ProbabilityEstimateNonconformity.Learner<>(pet));
    }

    @Override
    public InductiveConformalClassifier<In, Out> fit(Input<In> in, List<Out> out) {
      Objects.requireNonNull(in, "Input data is required.");
      Objects.requireNonNull(out, "Input target is required.");
      Check.argument(in.size() == out.size(),
          "The size of input data and input target don't match.");
      Nonconformity<In, Out> nc = learner.fit(in, out);
      boolean stochasticSmoothing = getOrDefault(STOCHASTIC_SMOOTHING);
      return new InductiveConformalClassifier<>(nc.getUniqueOutputs(), nc, calibrator,
          stochasticSmoothing);
    }
  }
}
