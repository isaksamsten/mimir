package org.mimirframework.classification.conformal;

import java.util.Collections;
import java.util.Objects;
import java.util.Set;

import org.briljantframework.Check;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.mimirframework.classification.ClassifierCharacteristic;
import org.mimirframework.supervised.Characteristic;
import org.mimirframework.supervised.Predictor;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class InductiveConformalClassifier extends AbstractConformalClassifier {

  private final ClassifierNonconformity nonconformity;
  private ClassifierCalibrator calibrator;
  private ClassifierCalibratorScores calibration = null;

  protected InductiveConformalClassifier(ClassifierNonconformity nonconformity,
      ClassifierCalibrator calibrator, boolean stochasticSmoothing, Vector classes) {
    super(stochasticSmoothing, classes);
    this.calibrator = Objects.requireNonNull(calibrator, "Calibrator is required.");
    this.nonconformity = Objects.requireNonNull(nonconformity, "Requires nonconformity scorer");
  }

  /**
   * Calibrate this inductive conformal classifier using the supplied data frame and output target
   *
   * @param x the data
   * @param y the calibration target
   */
  public void calibrate(DataFrame x, Vector y) {
    calibration = calibrator.calibrate(nonconformity, x, y);
  }

  public DoubleArray getCalibrationScore(Vector example, Object label) {
    return calibration.get(example, label);
  }

  public ClassifierNonconformity getClassifierNonconformity() {
    return nonconformity;
  }

  @Override
  protected ClassifierCalibratorScores getCalibrationScores() {
    Check.state(calibration != null, "Classifier is not calibrated.");
    return calibration;
  }

  @Override
  public Set<Characteristic> getCharacteristics() {
    return Collections.singleton(ClassifierCharacteristic.ESTIMATOR);
  }

  /**
   * @author Isak Karlsson <isak-kar@dsv.su.se>
   */
  public static class Learner implements Predictor.Learner<InductiveConformalClassifier> {

    private final ClassifierNonconformity.Learner learner;
    private final boolean stochasticSmoothing;
    private final ClassifierCalibrator calibratable;

    public Learner(ClassifierNonconformity.Learner learner, ClassifierCalibrator calibrator,
        boolean stochasticSmoothing) {
      this.stochasticSmoothing = stochasticSmoothing;
      this.calibratable = Objects.requireNonNull(calibrator, "Calibrator is required.");
      this.learner = Objects.requireNonNull(learner, "Nonconformity learner is required.");
    }

    public Learner(ClassifierNonconformity.Learner learner, ClassifierCalibrator calibrator) {
      this(learner, calibrator, true);
    }

    public Learner(ClassifierNonconformity.Learner learner) {
      this(learner, ClassifierCalibrator.unconditional());
    }

    @Override
    public InductiveConformalClassifier fit(DataFrame x, Vector y) {
      Objects.requireNonNull(x, "Input data is required.");
      Objects.requireNonNull(y, "Input target is required.");
      Check.argument(x.rows() == y.size(), "The size of input data and input target don't match.");
      ClassifierNonconformity nc = learner.fit(x, y);
      return new InductiveConformalClassifier(nc, calibratable, stochasticSmoothing,
          nc.getClasses());
    }

  }
}
