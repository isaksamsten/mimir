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

import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.classification.Ensemble;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Output;
import org.briljantframework.mimir.supervised.AbstractLearner;

/**
 * @author Isak Karlsson
 */
public class BootstrapConformalClassifier<In> extends AbstractConformalClassifier<In> {

  private final ClassifierCalibratorScores<In> calibratorScores;
  private final ClassifierNonconformity<In> nonconformity;

  private BootstrapConformalClassifier(boolean stochasticSmoothing, List<?> classes,
      ClassifierCalibratorScores<In> calibratorScores, ClassifierNonconformity<In> nonconformity) {
    super(stochasticSmoothing, classes);
    this.calibratorScores = calibratorScores;
    this.nonconformity = nonconformity;
  }

  @Override
  protected ClassifierNonconformity<In> getClassifierNonconformity() {
    return nonconformity;
  }

  @Override
  protected ClassifierCalibratorScores<In> getCalibrationScores() {
    return calibratorScores;
  }

  public static class Learner<In>
      extends AbstractLearner<In, Object, BootstrapConformalClassifier<In>> {

    private final ProbabilityEstimateNonconformity.Learner<In, ? extends Ensemble<In>> learner;

    public Learner(ProbabilityEstimateNonconformity.Learner<In, ? extends Ensemble<In>> learner) {
      this.learner = learner;
    }

    @Override
    public BootstrapConformalClassifier<In> fit(Input<? extends In> x, Output<?> y) {
      ProbabilityEstimateNonconformity<In, ? extends Ensemble<In>> pen = learner.fit(x, y);
      Ensemble<In> ensemble = pen.getClassifier();
      DoubleArray estimate = Ensemble.estimateOutOfBagProbabilities(ensemble, x);
      DoubleArray calibrationScores = ProbabilityCostFunction
          .estimate(pen.getProbabilityCostFunction(), estimate, y, ensemble.getClasses());
      return new BootstrapConformalClassifier<>(getOrDefault(STOCHASTIC_SMOOTHING),
          ensemble.getClasses(), (example, label) -> calibrationScores, pen);
    }
  }

}
