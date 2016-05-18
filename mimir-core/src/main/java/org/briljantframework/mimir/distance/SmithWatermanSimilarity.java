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

import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.series.Series;

/**
 * @author Isak Karlsson
 */
public class SmithWatermanSimilarity implements Similarity {

  private final double match, miss, gap;

  public SmithWatermanSimilarity(double match, double miss, double gap) {
    this.match = match;
    this.miss = miss;
    this.gap = gap;
  }

  @Override
  public double compute(Series a, Series b) {
    DoubleArray h = DoubleArray.zeros(a.size() + 1, b.size() + 1);
    double maxScore = Double.NEGATIVE_INFINITY;
    for (int i = 1; i < h.rows(); i++) {
      for (int j = 1; j < h.columns(); j++) {
        double sim = h.get(i - 1, j - 1) + (a.loc().equals(i - 1, b, j - 1) ? match : miss);
        double left = h.get(i, j - 1) + gap;
        double up = h.get(i - 1, j) + gap;
        double score = Math.max(0, Math.max(sim, Math.max(up, left)));
        h.set(i, j, score);
        if (score > maxScore) {
          maxScore = score;
        }
      }
    }
    return maxScore;
  }
}
