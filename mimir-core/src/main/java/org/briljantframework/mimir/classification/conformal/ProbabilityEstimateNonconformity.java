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

import java.util.List;
import java.util.Objects;

import org.briljantframework.Check;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.Input;
import org.briljantframework.mimir.Output;
import org.briljantframework.mimir.classification.Classifier;

/**
 * Implements a nonconformity scorer based on an underlying classifier and a probability cost
 * function.
 * 
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ProbabilityEstimateNonconformity<In, T extends Classifier<In>>
    implements ClassifierNonconformity<In, Object> {

  private final T classifier;
  private final ProbabilityCostFunction probabilityCostFunction;

  public ProbabilityEstimateNonconformity(T classifier,
      ProbabilityCostFunction probabilityCostFunction) {
    this.classifier = Objects.requireNonNull(classifier);
    this.probabilityCostFunction = Objects.requireNonNull(probabilityCostFunction);
  }

  /**
   * Returns the classifier used as probability estimator
   * 
   * @return the classifier used as probability estimator
   */
  public T getClassifier() {
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
  public DoubleArray estimate(Input<? extends In> x, Output<?> y) {
    Objects.requireNonNull(x, "Input data required.");
    Objects.requireNonNull(y, "Input target required.");
    Check.argument(x.size() == y.size(), "The size of input data and input target don't match.");
    DoubleArray estimate = getClassifier().estimate(x);
    return ProbabilityCostFunction.estimate(getProbabilityCostFunction(), estimate, y,
        getClasses());
  }

  @Override
  public double estimate(In example, Object label) {
    Objects.requireNonNull(example, "Require an example.");
    int trueClassIndex = getClasses().indexOf(label);
    if (trueClassIndex < 0) {
      return 0;
    } else {
      return getProbabilityCostFunction().apply(getClassifier().estimate(example), trueClassIndex);
    }
  }

  @Override
  public List<?> getClasses() {
    return getClassifier().getClasses();
  }

  /**
   * @author Isak Karlsson <isak-kar@dsv.su.se>
   */
  public static class Learner<In, T extends Classifier<In>> implements
      ClassifierNonconformity.Learner<In, Object, ProbabilityEstimateNonconformity<In, T>> {

    private final Classifier.Learner<In, Object, ? extends T> classifier;
    private final ProbabilityCostFunction probabilityCostFunction;

    /**
     * Constructs a probability nonconformity learner
     *
     * @param classifier the classifier
     * @param probabilityCostFunction the probability cost function
     */
    public Learner(Classifier.Learner<In, Object, ? extends T> classifier,
        ProbabilityCostFunction probabilityCostFunction) {
      this.classifier = Objects.requireNonNull(classifier, "A classifier is required.");
      this.probabilityCostFunction =
          Objects.requireNonNull(probabilityCostFunction, "A cost function is required");
    }

    /**
     * Returns the classifier learner.
     * 
     * @return the classifier learner
     */
    protected Classifier.Learner<In, Object, ? extends T> getClassifierLearner() {
      return classifier;
    }

    /**
     * Returns the probability cost function.
     * 
     * @return the probability cost function
     */
    protected ProbabilityCostFunction getProbabilityCostFunction() {
      return probabilityCostFunction;
    }

    @Override
    public ProbabilityEstimateNonconformity<In, T> fit(Input<? extends In> x, Output<?> y) {
      Objects.requireNonNull(x, "Input data is required.");
      Objects.requireNonNull(y, "Input target is required.");
      Check.argument(x.size() == y.size(), "The size of input data and input target don't match");
      T pe = getClassifierLearner().fit(x, y);
      Check.state(pe != null && pe.getCharacteristics().contains(ESTIMATOR),
          "The produced classifier can't estimate probabilities");
      return new ProbabilityEstimateNonconformity<>(pe, getProbabilityCostFunction());
    }
  }
}
