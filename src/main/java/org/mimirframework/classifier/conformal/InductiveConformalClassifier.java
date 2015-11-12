package org.mimirframework.classifier.conformal;

import java.util.Collections;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

import org.briljantframework.Check;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.data.vector.Vectors;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class InductiveConformalClassifier extends org.mimirframework.classification.AbstractClassifier implements ConformalClassifier {

  private final Nonconformity nonconformity;

  /**
   * [no-calibration, 1] double array of nonconformity scores for the calibration set used to
   * estimate the p-values when performing a conformal prediction
   */
  private DoubleArray calibration;

  protected InductiveConformalClassifier(Nonconformity nonconformity, Vector classes) {
    super(classes);
    this.nonconformity = Objects.requireNonNull(nonconformity, "Requires nonconformity scorer");
  }

  @Override
  public void calibrate(DataFrame x, Vector y) {
    calibration = nonconformity.estimate(x, y);
  }

  @Override
  public DoubleArray estimate(Vector example) {
    Check.state(calibration != null, "the conformal predictor must be calibrated");
    DoubleArray significance = Arrays.newDoubleArray(getClasses().size());
    double n = calibration.size();
    for (int i = 0; i < significance.size(); i++) {
      Object label = getClasses().loc().get(i);
      double nc = nonconformity.estimate(example, label);
      double gt = calibration.filter(score -> score >= nc).size();
      significance.set(i, (gt + 1) / (n + 1));
    }
    return significance;
  }

  @Override
  public Set<org.mimirframework.supervised.Characteristic> getCharacteristics() {
    return new HashSet<>(Collections.singletonList(org.mimirframework.classification.ClassifierCharacteristic.ESTIMATOR));
  }

  /**
   * @author Isak Karlsson <isak-kar@dsv.su.se>
   */
  public static class Learner implements org.mimirframework.supervised.Predictor.Learner<InductiveConformalClassifier> {

    private final Nonconformity.Learner learner;

    public Learner(Nonconformity.Learner learner) {
      this.learner = Objects.requireNonNull(learner);
    }

    @Override
    public InductiveConformalClassifier fit(DataFrame x, Vector y) {
      Objects.requireNonNull(x, "Input data is required.");
      Objects.requireNonNull(y, "Input target is required.");
      Check.argument(x.rows() == y.size(), "The size of input data and input target don't match.");
      return new InductiveConformalClassifier(learner.fit(x, y), Vectors.unique(y));
    }

  }
}
