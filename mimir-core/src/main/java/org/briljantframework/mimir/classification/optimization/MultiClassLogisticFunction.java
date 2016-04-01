/**
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Isak Karlsson
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
 * associated documentation files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package org.briljantframework.mimir.classification.optimization;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.array.IntArray;
import org.briljantframework.optimize.DifferentialMultivariateFunction;

/**
 * @author Isak Karlsson
 */
public class MultiClassLogisticFunction implements DifferentialMultivariateFunction {

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
}
