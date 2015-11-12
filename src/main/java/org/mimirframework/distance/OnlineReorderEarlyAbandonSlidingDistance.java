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

import org.briljantframework.data.vector.Vector;
import org.briljantframework.data.vector.Vectors;
import org.mimirframework.shapelet.IndexSortedNormalizedShapelet;

/**
 * TODO(isak): make this work! Created by Isak Karlsson on 27/10/14.
 */
public class OnlineReorderEarlyAbandonSlidingDistance extends EarlyAbandonSlidingDistance {

  /**
   * Instantiates a new Online reorder early abandon sliding distance.
   *
   * @param distance the distance
   */
  public OnlineReorderEarlyAbandonSlidingDistance(Distance distance) {
    super(distance);
  }

  /**
   * Create online reorder early abandon sliding distance.
   *
   * @return the online reorder early abandon sliding distance
   */
  public static OnlineReorderEarlyAbandonSlidingDistance create() {
    return new OnlineReorderEarlyAbandonSlidingDistance(Euclidean.getInstance());
  }


  @Override
  public double compute(Vector a, Vector b) {
    // Candidate is normalized
    Vector candidate = a.size() < b.size() ? a : b;
    int l = candidate.size();

    int[] order;
    if (candidate instanceof IndexSortedNormalizedShapelet) {
      order = ((IndexSortedNormalizedShapelet) candidate).getSortOrder();
    } else {
      order =
          Vectors.indexSort(
              candidate,
              (i, j) -> Double.compare(Math.abs(candidate.loc().getAsDouble(j)),
                  Math.abs(candidate.loc().getAsDouble(i))));
    }

    Vector vector = a.size() >= b.size() ? a : b;
    int m = vector.size();

    double ex = 0, ex2 = 0; // running sum and square sum
    double best = Double.POSITIVE_INFINITY;

    // todo: get this temporary array out of the way
    double[] T = new double[l * 2];
    int loc = 0;

    for (int i = 0; i < m; i++) {
      double ti = vector.loc().getAsDouble(i);
      T[i % l] = ti;
      T[(i % l) + l] = ti;

      ex += ti;
      ex2 += ti * ti;

      if (i >= l - 1) {
        double mean = ex / l;
        double std = Math.sqrt(ex2 / l - (mean * mean));
        int index = (i + 1) % l;

        int j = 0;
        double d = 0;
        while (j < l && d < best) {
          double sum = candidate.loc().getAsDouble(order[j]) - (T[order[j]] - mean) / std;
          d += sum * sum;
          j++;
        }

        if (j == l && d < best) {
          loc = i - l + 1;
          best = d;
        }
        double v = T[index];
        ex -= v;
        ex2 -= v * v;
      }

    }
    return best;
  }
}
