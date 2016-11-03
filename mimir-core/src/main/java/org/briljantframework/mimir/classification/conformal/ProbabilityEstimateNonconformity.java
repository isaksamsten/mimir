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

import static org.briljantframework.mimir.classification.ClassifierCharacteristic.ESTIMATOR;

import java.util.Objects;

import org.briljantframework.Check;
import org.briljantframework.array.Array;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.classification.ProbabilityEstimator;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Output;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * Implements a nonconformity scorer based on an underlying classifier and a probability cost
 * function.
 *
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ProbabilityEstimateNonconformity<In, Out> implements Nonconformity<In, Out> {

  private final ProbabilityEstimator<In, Out> classifier;
  private final ProbabilityCostFunction probabilityCostFunction;

  public ProbabilityEstimateNonconformity(ProbabilityEstimator<In, Out> classifier,
      ProbabilityCostFunction probabilityCostFunction) {
    this.classifier = Objects.requireNonNull(classifier);
    this.probabilityCostFunction = Objects.requireNonNull(probabilityCostFunction);
  }

  /**
   * Returns the classifier used as probability estimator
   *
   * @return the classifier used as probability estimator
   */
  public ProbabilityEstimator<In, Out> getProbabilityEstimator() {
    return classifier;
  }

  /**
   * Returns the error function
   *
   * @return the error function
   */
  public ProbabilityCostFunction getProbabilityCostFunction() {
    return probabilityCostFunction;
  }

  @Override
  public DoubleArray estimate(Input<? extends In> x, Output<? extends Out> y) {
    Objects.requireNonNull(x, "Input data required.");
    Objects.requireNonNull(y, "Input target required.");
    Check.argument(x.size() == y.size(), "The size of input data and input target don't match.");
    DoubleArray estimate = getProbabilityEstimator().estimate(x);
    return ProbabilityCostFunction.estimate(getProbabilityCostFunction(), estimate, y,
        getProbabilityEstimator().getClasses());
  }

  @Override
  public double estimate(In example, Out label) {
    Objects.requireNonNull(example, "Require an example.");
    int trueClassIndex = classifier.getClasses().indexOf(label);
    if (trueClassIndex < 0) {
      return 0;
    } else {
      return getProbabilityCostFunction().apply(getProbabilityEstimator().estimate(example),
          trueClassIndex);
    }
  }

  @Override
  public Array<Out> getUniqueOutputs() {
    return getProbabilityEstimator().getClasses();
  }

  /**
   * @author Isak Karlsson <isak-kar@dsv.su.se>
   */
  public static class Learner<In, Out> implements Nonconformity.Learner<In, Out> {

    private final Predictor.Learner<In, Out, ? extends ProbabilityEstimator<In, Out>> probabilityEstimator;
    private final ProbabilityCostFunction probabilityCostFunction;

    /**
     * Constructs a probability nonconformity learner
     *
     * @param probabilityEstimator the classifier
     * @param probabilityCostFunction the probability cost function
     */
    public Learner(
        Predictor.Learner<In, Out, ? extends ProbabilityEstimator<In, Out>> probabilityEstimator,
        ProbabilityCostFunction probabilityCostFunction) {
      this.probabilityEstimator =
          Objects.requireNonNull(probabilityEstimator, "A classifier is required.");
      this.probabilityCostFunction =
          Objects.requireNonNull(probabilityCostFunction, "A cost function is required");
    }

    public Learner(Predictor.Learner<In, Out, ? extends ProbabilityEstimator<In, Out>> pet) {
      this(pet, ProbabilityCostFunction.margin());
    }

    @Override
    public ProbabilityEstimateNonconformity<In, Out> fit(Input<? extends In> x,
        Output<? extends Out> y) {
      Objects.requireNonNull(x, "Input data is required.");
      Objects.requireNonNull(y, "Input target is required.");
      Check.argument(x.size() == y.size(), "The size of input data and input target don't match");
      ProbabilityEstimator<In, Out> pe = probabilityEstimator.fit(x, y);
      Check.state(pe != null && pe.getCharacteristics().contains(ESTIMATOR),
          "The produced classifier can't estimate probabilities");
      return new ProbabilityEstimateNonconformity<>(pe, probabilityCostFunction);
    }
  }
}
