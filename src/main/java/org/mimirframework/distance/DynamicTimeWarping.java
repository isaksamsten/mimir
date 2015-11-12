/*
 * The MIT License (MIT)
 * 
 * Copyright (c) 2015 Isak Karlsson
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

package org.mimirframework.distance;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.vector.Vector;

/**
 * In time series analysis, dynamic time warping (DTW) is an algorithm for measuring similarity
 * between two temporal sequences which may vary in time or speed.
 * <p>
 * In general, DTW is a method that calculates an optimal match between two given sequences (e.g.
 * time series) with certain restrictions. The sequences are "warped" non-linearly in the time
 * dimension to determine a measure of their similarity independent of certain non-linear variations
 * in the time dimension. This sequence alignment method is often used in time series
 * classification. Although DTW measures a distance-like quantity between two given sequences, it
 * doesn't guarantee the triangle inequality to hold.
 * <p>
 * Created by Isak Karlsson on 01/09/14.
 */
public class DynamicTimeWarping implements Distance {

  /**
   * The Distance.
   */
  protected final Distance distance;
  private final int constraint;

  /**
   * Instantiates a new Dynamic time warping.
   *
   * @param distance the local distance function
   * @param constraint the local constraint (i.e. width of the band)
   */
  public DynamicTimeWarping(Distance distance, int constraint) {
    this.distance = distance;
    this.constraint = constraint;
  }

  public DynamicTimeWarping(int constraint) {
    this(EuclideanDistance.getInstance(), constraint);
  }

  /**
   * Delegated to the injected distance function
   *
   * @param a scalar
   * @param b scalar
   * @return the distance between a and b
   */
  @Override
  public final double compute(double a, double b) {
    return distance.compute(a, b);
  }

  @Override
  public double compute(Vector a, Vector b) {
    int n = a.size(), m = b.size();
    DoubleArray dtw = Arrays.newDoubleArray(n, m);
    dtw.assign(Double.POSITIVE_INFINITY);
    dtw.set(0, 0, 0);

    int width = Math.max(constraint, Math.abs(n - m));
    for (int i = 1; i < n; i++) {
      int end = constraint <= -1 ? m : Math.min(m, i + width);
      int start = constraint <= -1 ? 1 : Math.max(1, i - width);
      for (int j = start; j < end; j++) {
        double cost = distance.compute(a.loc().getAsDouble(i), b.loc().getAsDouble(j));
        dtw.set(i, j,
            cost + Math.min(dtw.get(i - 1, j), Math.min(dtw.get(i, j - 1), dtw.get(i - 1, j - 1))));
      }
    }

    return dtw.get(n - 1, m - 1);
  }

  @Override
  public double max() {
    return distance.max();
  }

  @Override
  public double min() {
    return distance.min();
  }

  @Override
  public String toString() {
    return String.format("Dynamic time warping (w=%s)", constraint);
  }
}
