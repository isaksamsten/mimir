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

    // Initialize the linear regression learner
    LinearRegression.Learner learner = new LinearRegression.Learner();

    // Fit the model
    LinearRegression regression = learner.fit(x, y);

    // Print the parameters
    System.out.println(regression.getParameters());
  }

  private static DoubleArray sinTarget(DoubleArray x) {
    DoubleArray x0 = x.getColumn(0);
    DoubleArray x1 = x.getColumn(1);
    return Arrays.signum(x0.minus(x1).plus(Arrays.sin(x0).times(Math.PI)));
  }
}
