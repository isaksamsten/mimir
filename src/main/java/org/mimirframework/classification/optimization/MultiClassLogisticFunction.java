package org.mimirframework.classification.optimization;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.IntArray;
import org.briljantframework.optimize.DifferentialFunction;

/**
 * @author Isak Karlsson
 */
public class MultiClassLogisticFunction implements DifferentialFunction {

  private final DoubleArray x;
  private final IntArray y;
  private final double lambda;
  private final int k;

  public MultiClassLogisticFunction(DoubleArray x, IntArray y, double lambda, int k) {
    this.x = x;
    this.y = y;
    this.lambda = lambda;
    this.k = k;
  }

  @Override
  public double gradientCost(DoubleArray w, DoubleArray g) {
    double f = 0.0;
    int n = x.rows();
    int p = x.columns();
    w = w.reshape(p, k);
    g = g.reshape(p, k);
    g.assign(0);
    DoubleArray prob = DoubleArray.zeros(k);
    for (int i = 0; i < n; i++) {
      DoubleArray xi = x.getRow(i);
      for (int j = 0; j < k; j++) {
        prob.set(j, Arrays.inner(xi, w.getColumn(j)));
      }
      OptimizationUtils.softMax(prob);
      f -= OptimizationUtils.stableLog(prob.get(y.get(i)));
      for (int j = 0; j < k; j++) {
        double yi = (y.get(i) == j ? 1.0 : 0.0) - prob.get(j);
        for (int l = 1; l < p; l++) {
          g.set(l, j, g.get(l, j) - yi * x.get(i, l));
        }
        g.set(0, j, g.get(0, j) - yi);
      }
    }
    return computeShrinkage(w, f, p);
  }

  @Override
  public double cost(DoubleArray w) {
    double f = 0.0;
    int n = x.rows();
    int p = x.columns();
    w = w.reshape(p, k);
    DoubleArray prob = DoubleArray.zeros(k);
    for (int i = 0; i < n; i++) {
      DoubleArray xi = x.getRow(i);
      for (int j = 0; j < k; j++) {
        prob.set(j, Arrays.inner(xi, w.getColumn(j)));
      }

      OptimizationUtils.softMax(prob);
      f -= OptimizationUtils.stableLog(prob.get(y.get(i)));
    }
    return computeShrinkage(w, f, p);
  }

  private double computeShrinkage(DoubleArray w, double f, int p) {
    if (lambda != 0.0) {
      double w2 = 0.0;
      for (int i = 0; i < k; i++) {
        for (int j = 0; j < p; j++) {
          double v = w.get(j, i);
          w2 += v * v;
        }
      }

      f += 0.5 * lambda * w2;
    }
    return f;
  }
}
