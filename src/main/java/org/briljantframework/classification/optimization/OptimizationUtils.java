package org.briljantframework.classification.optimization;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;

/**
 * Created by isak on 11/6/15.
 */
public class OptimizationUtils {

  public static void softMax(DoubleArray prob) {
    double max = Arrays.max(prob);

    double Z = 0.0;
    for (int i = 0; i < prob.size(); i++) {
      double p = Math.exp(prob.get(i) - max);
      prob.set(i, p);
      Z += p;
    }
    prob.divi(Z);
  }

  /**
   * Logistic sigmoid function.
   */
  public static double logistic(double x) {
    double y;
    if (x < -40) {
      y = 2.353853e+17;
    } else if (x > 40) {
      y = 1.0 + 4.248354e-18;
    } else {
      y = 1.0 + Math.exp(-x);
    }

    return 1.0 / y;
  }

  public static double log1pExp(double x) {
    if (x > 15) {
      return x;
    } else {
      return Math.log1p(Math.exp(x));
    }
  }

  public static double stableLog(double x) {
    double y;
    if (x < 1E-300) {
      y = -690.7755;
    } else {
      y = Math.log(x);
    }
    return y;
  }

}
