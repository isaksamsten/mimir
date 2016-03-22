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
package org.briljantframework.mimir.distance;


import org.briljantframework.DoubleSequence;

/**
 * EuclideanDistance between two Points (either Vectors or scalars)
 * <p>
 * Created by Isak Karlsson on 01/09/14.
 */
public class EuclideanDistance implements Distance<DoubleSequence> {

  private static final EuclideanDistance instance = new EuclideanDistance();

  private EuclideanDistance() {

  }

  /**
   * The constant instance.
   */
  public static EuclideanDistance getInstance() {
    return instance;
  }

  private double compute(double a, double b) {
    double r = a - b;
    return r * r;
  }

  @Override
  public double compute(DoubleSequence a, DoubleSequence b) {
    int size = Math.min(a.size(), b.size());

    double residual = 0.0;
    for (int i = 0; i < size; i++) {
      residual += compute(a.getAsDouble(i), b.getAsDouble(i));
    }

    return Math.sqrt(residual);
  }

  @Override
  public String toString() {
    return "EuclideanDistance";
  }
}
