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

import java.util.Objects;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.mimir.classification.Ensemble;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * @author Isak Karlsson
 */
public class BootstrapConformalClassifier extends AbstractConformalClassifier {

  private final ClassifierCalibratorScores calibratorScores;
  private final ClassifierNonconformity nonconformity;

  protected BootstrapConformalClassifier(ClassifierCalibratorScores calibratorScores,
      ClassifierNonconformity nonconformity, Vector classes) {
    super(true, classes);
    this.calibratorScores = Objects.requireNonNull(calibratorScores);
    this.nonconformity = Objects.requireNonNull(nonconformity);
  }

  @Override
  protected ClassifierNonconformity getClassifierNonconformity() {
    return nonconformity;
  }

  @Override
  protected ClassifierCalibratorScores getCalibrationScores() {
    return calibratorScores;
  }

  public static class Learner implements Predictor.Learner<BootstrapConformalClassifier> {

    private final ProbabilityEstimateNonconformity.Learner<? extends Ensemble> learner;

    public Learner(ProbabilityEstimateNonconformity.Learner<? extends Ensemble> learner) {
      this.learner = learner;
    }

    @Override
    public BootstrapConformalClassifier fit(DataFrame x, Vector y) {
      ProbabilityEstimateNonconformity<? extends Ensemble> pen = learner.fit(x, y);
      Ensemble ensemble = pen.getClassifier();
      DoubleArray estimate = Ensemble.oobEstimates(ensemble, x);
      DoubleArray calibrationScores =
          ProbabilityCostFunction.estimate(pen.getProbabilityCostFunction(), estimate, y,
              ensemble.getClasses());
      return new BootstrapConformalClassifier((example, label) -> calibrationScores, pen,
          ensemble.getClasses());
    }
  }

}
