package org.mimirframework.classification.conformal;

import java.util.Objects;

import org.briljantframework.Check;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.mimirframework.classification.Classifier;
import org.mimirframework.classification.ClassifierCharacteristic;
import org.mimirframework.supervised.Predictor;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ProbabilityEstimateNonconformity implements Nonconformity {

  private final Classifier classifier;
  private final ProbabilityCostFunction errorFunction;

  ProbabilityEstimateNonconformity(Classifier classifier, ProbabilityCostFunction errorFunction) {
    this.classifier = classifier;
    this.errorFunction = errorFunction;
  }

  @Override
  public double estimate(Vector example, Object label) {
    Objects.requireNonNull(example, "Require an example.");
    int trueClassIndex = classifier.getClasses().loc().indexOf(label);
    Check.argument(trueClassIndex >= 0, "illegal label %s", label);
    return errorFunction.apply(classifier.estimate(example), trueClassIndex);
  }

  @Override
  public DoubleArray estimate(DataFrame x, Vector y) {
    Objects.requireNonNull(x, "Input data required.");
    Objects.requireNonNull(y, "Input target required.");
    Check.argument(x.rows() == y.size(), "The size of input data and input target don't match.");
    return errorFunction.apply(classifier.estimate(x), y, classifier.getClasses());
  }

  /**
   * @author Isak Karlsson <isak-kar@dsv.su.se>
   */
  public static class Learner implements Nonconformity.Learner {

    private final Predictor.Learner<? extends Classifier> classifier;
    private final ProbabilityCostFunction errorFunction;

    public Learner(Predictor.Learner<? extends Classifier> classifier,
                   ProbabilityCostFunction errorFunction) {
      this.classifier = Objects.requireNonNull(classifier, "A classifier is required.");
      this.errorFunction = Objects.requireNonNull(errorFunction, "An error function is required");

    }

    @Override
    public Nonconformity fit(DataFrame x, Vector y) {
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
