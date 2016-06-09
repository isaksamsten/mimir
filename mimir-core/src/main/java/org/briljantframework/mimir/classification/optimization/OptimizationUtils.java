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

/**
 * @author Isak Karlsson
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
    Arrays.scal(1 / Z, prob);
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
