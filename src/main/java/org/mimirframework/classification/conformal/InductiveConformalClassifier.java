package org.mimirframework.classification.conformal;

import java.util.Collections;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

import org.briljantframework.Check;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.data.vector.Vectors;
import org.mimirframework.classification.ClassifierCharacteristic;
import org.mimirframework.supervised.Characteristic;
import org.mimirframework.supervised.Predictor;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class InductiveConformalClassifier extends AbstractConformalClassifier {

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

  /**
   * Calibrate this inductive conformal classifier using the supplied dataframe and output target
   * 
   * @param x the ddata
   * @param y the calibration target
   */
  public void calibrate(DataFrame x, Vector y) {
    calibration = nonconformity.estimate(x, y);
  }

  public DoubleArray getCalibration() {
    return calibration;
  }

  public Nonconformity getNonconformity() {
    return nonconformity;
  }

  @Override
  public Set<Characteristic> getCharacteristics() {
    return new HashSet<>(Collections.singletonList(ClassifierCharacteristic.ESTIMATOR));
  }

  /**
   * @author Isak Karlsson <isak-kar@dsv.su.se>
   */
  public static class Learner implements Predictor.Learner<InductiveConformalClassifier> {

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
