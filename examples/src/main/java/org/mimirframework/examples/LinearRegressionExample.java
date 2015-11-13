package org.mimirframework.examples;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.mimirframework.regression.LinearRegression;

/**
 * @author Isak Karlsson
 */
public class LinearRegressionExample {
  public static void main(String[] args) {
    DoubleArray x = Arrays.randn(100).reshape(50, 2);
    DoubleArray y = sinTarget(x);

    LinearRegression.Learner learner = new LinearRegression.Learner();
    LinearRegression regression = learner.fit(x, y);
    System.out.println(regression.getTheta());
  }

  private static DoubleArray sinTarget(DoubleArray x) {
    return x.getColumn(0).minus(x.getColumn(1)).plus(x.getColumn(0).map(i -> Math.sin(i) * Math.PI))
        .map(Math::signum);
  }
}
