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

package org.briljantframework.distance;

import java.util.Objects;

import org.briljantframework.data.vector.Vector;
import org.briljantframework.shapelet.IndexSortedNormalizedShapelet;
import org.briljantframework.shapelet.NormalizedShapelet;

/**
 * Created by Isak Karlsson on 23/09/14.
 */
public class EarlyAbandonSlidingDistance implements Distance {

  protected final Distance distance;

  /**
   * Instantiates a new Early abandon sliding distance.
   *
   * @param distance the distance
   */
  public EarlyAbandonSlidingDistance(Distance distance) {
    this.distance = Objects.requireNonNull(distance, "Requires a distance measure");
  }

  /**
   * Create early abandon sliding distance.
   *
   * @param distance the distance
   * @return the early abandon sliding distance
   */
  public static EarlyAbandonSlidingDistance create(Distance distance) {
    return new EarlyAbandonSlidingDistance(distance);
  }

  @Override
  public double compute(double a, double b) {
    return distance.compute(a, b);
  }

  /**
   * If {@code a} is shorter than {@code b}, then {@code a} is considered a shapelet and slid
   * against {@code b} and wise-versa.
   *
   * <p>
   * The shorter vector (i.e. the shapelet) is expected to be z-normalized
   *
   * @param a a vector
   * @param b a vector
   * @return the shortest possible distance of a (or b) as it is slid against b (or a)
   */
  @Override
  public double compute(Vector a, Vector b) {
    double minDistance = Double.POSITIVE_INFINITY;
    Vector candidate = a.size() < b.size() ? a : b;
    Vector vector = a.size() >= b.size() ? a : b;
    if (!(candidate instanceof NormalizedShapelet)) {
      if (!(vector instanceof NormalizedShapelet)) {
        vector = new NormalizedShapelet(0, vector.size(), vector);
      }
      return new SlidingDistance(Euclidean.getInstance()).compute(vector, new NormalizedShapelet(0,
                                                                                                 candidate
                                                                                                     .size(),
                                                                                                 candidate));
      // candidate = new NormalizedShapelet(0, candidate.size(), candidate);
      // throw new IllegalArgumentException("Candidate shapelet must be z-normalized");
    }

    int[] order = null;
    // If the candidate is IndexSorted, use this to optimize the search
    if (candidate instanceof IndexSortedNormalizedShapelet) {
      order = ((IndexSortedNormalizedShapelet) candidate).getSortOrder();
    }

    int seriesSize = vector.size();
    int m = candidate.size();
    double[] t = new double[m * 2];

    double ex = 0;
    double ex2 = 0;
    for (int i = 0; i < seriesSize; i++) {
      double d = vector.loc().getAsDouble(i);
      ex += d;
      ex2 += d * d;
      t[i % m] = d;
      t[(i % m) + m] = d;

      if (i >= m - 1) {
        int j = (i + 1) % m;
        double mean = ex / m;
        double sigma = StrictMath.sqrt(ex2 / m - mean * mean);
        double dist = distance(candidate, t, j, m, order, mean, sigma, minDistance);
        if (dist < minDistance) {
          minDistance = dist;
        }

        ex -= t[j];
        ex2 -= t[j] * t[j];
      }
    }
    return Math.sqrt(minDistance / candidate.size());
  }

  @Override
  public double max() {
    return distance.max();
  }

  @Override
  public double min() {
    return distance.min();
  }

  double distance(Vector c, double[] t, int j, int m, int[] order, double mean, double std,
      double bsf) {
    double sum = 0;
    for (int i = 0; i < m && sum < bsf; i++) {
      if (order != null) {
        i = order[i];
      }
      double x = normalize(t[i + j], mean, std) - c.loc().getAsDouble(i);
      // double x = ((t[i + j] - mean) / std) - c.loc().getAsDouble(i);
      sum += x * x;
    }
    return sum;
  }

  public double normalize(double value, double mean, double std) {
    if (std == 0) {
      return 0;
    } else {
      return (value - mean) / std;
    }
  }
}
