package org.mimirframework.regression;

import java.util.Collections;
import java.util.Set;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public final class LinearRegression implements Regression {

  private final DoubleArray theta;

  private LinearRegression(DoubleArray theta) {
    this.theta = theta;
  }

  public DoubleArray getTheta() {
    return theta;
  }

  @Override
  public double predict(Vector y) {
    return 0;
  }

  @Override
  public Vector predict(DataFrame x) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Set<org.mimirframework.supervised.Characteristic> getCharacteristics() {
    return Collections.emptySet();
  }

  /**
   * @author Isak Karlsson
   */
  public static class Learner implements Regression.Learner<LinearRegression> {

    public Learner() {}

    @Override
    public LinearRegression fit(DoubleArray x, DoubleArray y) {
      x = Arrays.hstack(Arrays.ones(x.rows()).reshape(x.rows(), 1), x);
      DoubleArray inner = Arrays.mmul(x.transpose(), x);
      DoubleArray v = Arrays.mmul(Arrays.linalg.pinv(inner), x.transpose());
      DoubleArray theta = Arrays.mmul(v, y);
      return new LinearRegression(theta);
    }

  }
}
