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

import org.briljantframework.array.Array;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.classification.Ensemble;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Output;
import org.briljantframework.mimir.supervised.AbstractLearner;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * @author Isak Karlsson
 */
public class BootstrapConformalClassifier<In, Out> extends AbstractConformalClassifier<In, Out> {

  private final CalibratorScores<In, Out> calibratorScores;
  private final Nonconformity<In, Out> nonconformity;

  private BootstrapConformalClassifier(boolean stochasticSmoothing, Array<Out> classes,
      CalibratorScores<In, Out> calibratorScores, Nonconformity<In, Out> nonconformity) {
    super(classes, stochasticSmoothing);
    this.calibratorScores = calibratorScores;
    this.nonconformity = nonconformity;
  }

  @Override
  protected Nonconformity<In, Out> getClassifierNonconformity() {
    return nonconformity;
  }

  @Override
  protected CalibratorScores<In, Out> getCalibrationScores() {
    return calibratorScores;
  }

  public static class Learner<In, Out>
      extends AbstractLearner<In, Out, BootstrapConformalClassifier<In, Out>> {

    private final ProbabilityEstimateNonconformity.Learner<In, Out> learner;

    public Learner(Predictor.Learner<In, Out, ? extends Ensemble<In, Out>> learner) {
      this.learner =
          new ProbabilityEstimateNonconformity.Learner<>(learner, ProbabilityCostFunction.margin());
    }

    @Override
    public BootstrapConformalClassifier<In, Out> fit(Input<? extends In> x,
        Output<? extends Out> y) {
      ProbabilityEstimateNonconformity<In, Out> pen = learner.fit(x, y);
      @SuppressWarnings("unchecked")
      Ensemble<In, Out> ensemble = (Ensemble<In, Out>) pen.getProbabilityEstimator();
      DoubleArray estimate = Ensemble.estimateOutOfBagProbabilities(ensemble, x);
      DoubleArray calibrationScores = ProbabilityCostFunction
          .estimate(pen.getProbabilityCostFunction(), estimate, y, ensemble.getClasses());
      return new BootstrapConformalClassifier<>(getOrDefault(STOCHASTIC_SMOOTHING),
          ensemble.getClasses(), (example, label) -> calibrationScores, pen);
    }
  }

}
