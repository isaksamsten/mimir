package org.mimirframework.classification.conformal;

import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.util.Precision;
import org.briljantframework.Check;
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

  protected AbstractConformalClassifier(Vector classes) {
    super(classes);
  }

  protected abstract ClassifierNonconformity getNonconformity();

  protected abstract DoubleArray getCalibration();

  @Override
  public DoubleArray estimate(Vector example) {
    Check.state(getCalibration() != null, "the conformal predictor must be calibrated");
    DoubleArray significance = DoubleArray.zeros(getClasses().size());
    DoubleArray calibration = getCalibration();
    double tau = ThreadLocalRandom.current().nextDouble();
    double n = calibration.size() + 1;
    for (int i = 0; i < significance.size(); i++) {
      Object label = getClasses().loc().get(i);
      double nc = getNonconformity().estimate(example, label);
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
