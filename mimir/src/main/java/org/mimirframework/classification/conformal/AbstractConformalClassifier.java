package org.mimirframework.classification.conformal;

import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.util.Precision;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.vector.Vector;
import org.mimirframework.classification.AbstractClassifier;

/**
 * Implements the basic p-value computation for (inductive) conformal predictors given the
 * 
 * @author Isak Karlsson
 */
public abstract class AbstractConformalClassifier extends AbstractClassifier implements
    ConformalClassifier {

  private final boolean stochasticSmoothing;

  protected AbstractConformalClassifier(boolean stochasticSmoothing, Vector classes) {
    super(classes);
    this.stochasticSmoothing = stochasticSmoothing;
  }

  protected AbstractConformalClassifier(Vector classes) {
    this(true, classes);
  }

  /**
   * Return the non conformity scorer
   * 
   * @return a classifier nonconformity
   */
  protected abstract ClassifierNonconformity getClassifierNonconformity();

  /**
   * Get the calibration
   * 
   * @return a classifier calibration
   */
  protected abstract ClassifierCalibratorScores getCalibrationScores();

  @Override
  public DoubleArray estimate(Vector example) {
    DoubleArray significance = DoubleArray.zeros(getClasses().size());
    double tau = stochasticSmoothing ? ThreadLocalRandom.current().nextDouble() : 1;
    for (int i = 0; i < significance.size(); i++) {
      Object label = getClasses().loc().get(i);
      DoubleArray calibration = getCalibrationScores().get(example, label);
      double n = calibration.size() + 1;
      double nc = getClassifierNonconformity().estimate(example, label);
      double gt = 0;
      double eq = 1;
      for (int j = 0; j < calibration.size(); j++) {
        double v = calibration.get(j);
        if (v > nc) {
          gt++;
        } else if (Precision.equals(v, nc, 10e-6)) {
          eq++;
        }
      }
      significance.set(i, (gt + eq * tau) / n);
    }
    return significance;
  }

}
