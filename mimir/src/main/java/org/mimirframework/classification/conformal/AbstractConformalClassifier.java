package org.mimirframework.classification.conformal;

import org.briljantframework.Check;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.vector.Vector;
import org.mimirframework.classification.AbstractClassifier;

/**
 * @author Isak Karlsson
 */
public abstract class AbstractConformalClassifier extends AbstractClassifier
    implements ConformalClassifier {

  protected AbstractConformalClassifier(Vector classes) {
    super(classes);
  }

  protected abstract Nonconformity getNonconformity();

  protected abstract DoubleArray getCalibration();

  @Override
  public DoubleArray estimate(Vector example) {
    Check.state(getCalibration() != null, "the conformal predictor must be calibrated");
    DoubleArray significance = DoubleArray.zeros(getClasses().size());
    double n = getCalibration().size();
    for (int i = 0; i < significance.size(); i++) {
      Object label = getClasses().loc().get(i);
      double nc = getNonconformity().estimate(example, label);
      double gt = getCalibration().filter(score -> score >= nc).size();
      significance.set(i, (gt + 1) / (n + 1));
    }
    return significance;
  }

}
