package org.mimirframework.classification.optimization;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.IntArray;
import org.briljantframework.optimize.DifferentialFunction;

/**
 * Function for optimizing the binary
 *
 * @author Isak Karlsson
 */
public class BinaryLogisticFunction implements DifferentialFunction {

  private final double lambda;
  private final DoubleArray x;
  private final IntArray y;

  public BinaryLogisticFunction(DoubleArray x, IntArray y, double lambda) {
    this.lambda = lambda;
    this.x = x;
    this.y = y;
  }

  @Override
  public double gradientCost(DoubleArray w, DoubleArray g) {
    int p = w.size();
    int n = x.rows();
    g.assign(0);
    double f = 0.0;
    for (int i = 0; i < n; i++) {
      double wx = Arrays.inner(x.getRow(i), w);
      f += OptimizationUtils.log1pExp(wx) - y.get(i) * wx;

      double yi = y.get(i) - OptimizationUtils.logistic(wx);
      for (int j = 1; j < p; j++) {
        g.set(j, g.get(j) - yi * x.get(i, j));
      }
      g.set(0, g.get(0) - yi);
    }
    if (lambda != 0.0) {
      double w2 = 0.0;
      for (int i = 1; i < p; i++) {
        double v = w.get(i);
        w2 += v * v;
      }

      f += 0.5 * lambda * w2;
      for (int j = 1; j < p; j++) {
        g.set(j, g.get(j) + lambda * w.get(j));
      }
    }

    return f;
  }

  @Override
  public double cost(DoubleArray w) {
    int n = x.rows();
    double f = 0.0;
    for (int i = 0; i < n; i++) {
      double wx = Arrays.inner(x.getRow(i), w);
      f += OptimizationUtils.log1pExp(wx) - y.get(i) * wx;
    }

    if (lambda != 0.0) {
      int p = w.size() - 1;
      double w2 = 0.0;
      for (int i = 0; i < p; i++) {
        double v = w.get(i);
        w2 += v * v;
      }
      f += 0.5 * lambda * w2;
    }
    return f;
  }
}
