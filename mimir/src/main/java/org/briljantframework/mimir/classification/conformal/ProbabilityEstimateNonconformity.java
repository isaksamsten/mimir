package org.briljantframework.mimir.classification.conformal;

import java.util.Objects;

import org.briljantframework.Check;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.mimir.classification.Classifier;
import org.briljantframework.mimir.classification.ClassifierCharacteristic;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * Implements a nonconformity scorer based on an underlying classifier and a probability cost
 * function.
 * 
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ProbabilityEstimateNonconformity implements ClassifierNonconformity {

  private final Classifier classifier;
  private final ProbabilityCostFunction errorFunction;

  public ProbabilityEstimateNonconformity(Classifier classifier,
      ProbabilityCostFunction errorFunction) {
    this.classifier = Objects.requireNonNull(classifier);
    this.errorFunction = Objects.requireNonNull(errorFunction);
  }

  @Override
  public double estimate(Vector example, Object label) {
    Objects.requireNonNull(example, "Require an example.");
    int trueClassIndex = getClasses().loc().indexOf(label);
    Check.argument(trueClassIndex >= 0, "illegal label %s", label);
    return errorFunction.apply(classifier.estimate(example), trueClassIndex);
  }

  @Override
  public DoubleArray estimate(DataFrame x, Vector y) {
    Objects.requireNonNull(x, "Input data required.");
    Objects.requireNonNull(y, "Input target required.");
    Check.argument(x.rows() == y.size(), "The size of input data and input target don't match.");
    DoubleArray estimate = classifier.estimate(x);
    return ProbabilityCostFunction.estimate(errorFunction, estimate, y, getClasses());
  }

  @Override
  public Vector getClasses() {
    return classifier.getClasses();
  }

  /**
   * @author Isak Karlsson <isak-kar@dsv.su.se>
   */
  public static class Learner implements ClassifierNonconformity.Learner {

    private final Predictor.Learner<? extends Classifier> classifier;
    private final ProbabilityCostFunction errorFunction;

    public Learner(Predictor.Learner<? extends Classifier> classifier,
        ProbabilityCostFunction errorFunction) {
      this.classifier = Objects.requireNonNull(classifier, "A classifier is required.");
      this.errorFunction = Objects.requireNonNull(errorFunction, "An error function is required");

    }

    @Override
    public ClassifierNonconformity fit(DataFrame x, Vector y) {
      Objects.requireNonNull(x, "Input data is required.");
      Objects.requireNonNull(y, "Input target is required.");
      Check.argument(x.rows() == y.size(), "The size of input data and input target don't match");
      Classifier probabilityEstimator = classifier.fit(x, y);
      Check.state(
          probabilityEstimator != null
              && probabilityEstimator.getCharacteristics().contains(
                  ClassifierCharacteristic.ESTIMATOR),
          "The produced classifier can't estimate probabilities");
      return new ProbabilityEstimateNonconformity(probabilityEstimator, errorFunction);
    }

  }
}
