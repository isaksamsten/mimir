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
package org.briljantframework.mimir.shapelet;

import org.apache.commons.math3.util.Precision;
import org.briljantframework.DoubleVector;
import org.briljantframework.mimir.timeseries.data.TimeSeries;
import org.briljantframework.util.primitive.ArrayAllocations;

import java.util.Arrays;

/**
 * Created by isak on 2017-06-12.
 */
public class ShapeletCounter {

  private final double maxEuclideanDist;

  public ShapeletCounter(double maxEuclideanDist) {
    this.maxEuclideanDist = maxEuclideanDist;
  }

  public ShapeletMatch numberOfMatches(TimeSeries a, Shapelet b) {
    return numberOfMatches(a, b, maxEuclideanDist);
  }

  public ShapeletMatch numberOfMatches(TimeSeries a, Shapelet b, double maxEuclideanDist) {
    int[] order = null;
    // If the candidate is IndexSorted, use this to optimize the search
//    if (b instanceof IndexSortedNormalizedShapelet) {
//      order = ((IndexSortedNormalizedShapelet) b).getSortOrder();
//    }

    double bsf = maxEuclideanDist * maxEuclideanDist;
    int matches = 0;
    int seriesSize = a.size();
    int m = b.size();
    double[] t = new double[m * 2];

    double ex = 0;
    double ex2 = 0;
    int[] startMatch = new int[3];
    double[] dists = new double[3];
    double[] allDists = new double[a.size()];
    for (int i = 0; i < seriesSize; i++) {
      double d = a.getDouble(i);
      ex += d;
      ex2 += d * d;
      t[i % m] = d;
      t[(i % m) + m] = d;

      if (i >= m - 1) {
        int j = (i + 1) % m;
        double mean = ex / m;
        double sigma = StrictMath.sqrt(ex2 / m - mean * mean);
        double dist = Math.sqrt(distance(b, t, j, m, order, mean, sigma, Double.POSITIVE_INFINITY));
        if (Precision.compareTo(dist, maxEuclideanDist, 0.0001) <= 0) {
          startMatch = ArrayAllocations.ensureCapacity(startMatch, matches + 1);
          dists = ArrayAllocations.ensureCapacity(dists, matches + 1);
          startMatch[matches] = (i +1)- m;
          dists[matches] = dist;
          matches++;
        }
        allDists[i] = dist;

        ex -= t[j];
        ex2 -= t[j] * t[j];
      }

    }
    return new ShapeletMatch(startMatch, dists, matches, Arrays.copyOf(allDists, seriesSize));
  }

  double distance(DoubleVector c, double[] t, int j, int m, int[] order, double mean, double std,
      double bsf) {
    double sum = 0;
    for (int i = 0; i < m && Precision.compareTo(sum, bsf, 0.0001) <= 0; i++) {
      if (order != null) {
        i = order[i];
      }
      double x = normalize(t[i + j], mean, std) - c.getDouble(i);
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
