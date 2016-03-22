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
package org.briljantframework.mimir.data.timeseries;

import java.util.Objects;

import org.briljantframework.data.vector.Type;
import org.briljantframework.data.vector.Vector;

/**
 * Implements the Least Triangle ... ... Data Series resampler found in [cite the thesis].
 *
 * The implementation assumes that the values are ordered in increasing order.
 *
 * @author Isak Karlsson
 */
public class LeastTriagleThreeBucketAggregator implements Aggregator {

  private final int threshold;

  public LeastTriagleThreeBucketAggregator(int threshold) {
    this.threshold = threshold;
  }

  @Override
  public Type getAggregatedType() {
    return Type.DOUBLE;
  }

  @Override
  public Vector.Builder partialAggregate(Vector in) {
    Objects.requireNonNull(in);
    if (in.size() < threshold || threshold == 0) {
      return in.newCopyBuilder();
    }
    Vector.Builder sampled = in.newBuilder();

    // Bucket size
    double every = (double) (in.size() - 2) / (threshold - 2);

    int a = 0;
    int nextA = 0;

    int[] indexes = new int[in.size()];
    for (int i = 0; i < in.size(); i++) {
      indexes[i] = i;
    }
    sampled.add(in, a);
    for (int i = 0; i < threshold - 2; i++) {
      int avgX = 0;
      double avgY = 0;
      int avgRangeStart = (int) Math.floor((i + 1) * every) + 1;
      int avgRangeEnd = Math.min((int) Math.floor((i + 2) * every) + 1, in.size());
      int rangeLength = avgRangeEnd - avgRangeStart;

      for (; avgRangeStart < avgRangeEnd; avgRangeStart++) {
        avgX += indexes[avgRangeStart];
        avgY += in.loc().getAsDouble(avgRangeStart);
      }
      avgX /= rangeLength;
      avgY /= rangeLength;

      int rangeOffset = (int) Math.floor(i * every) + 1;
      int rangeTo = (int) Math.floor((i + 1) * every) + 1;

      int pointX = indexes[a];
      double pointY = in.loc().getAsDouble(a);
      double maxArea = Double.NEGATIVE_INFINITY;
      for (; rangeOffset < rangeTo; rangeOffset++) {
        double xDiff = pointX - avgX * (in.loc().getAsDouble(rangeOffset) - pointY);
        double yDiff = pointY - avgY * (pointX - indexes[rangeOffset]);
        double area = Math.abs(xDiff - yDiff) * 0.5;
        if (area > maxArea) {
          maxArea = area;
          nextA = rangeOffset;
        }
      }
      sampled.add(in, nextA);
      a = nextA;
    }

    return sampled.add(in, in.size() - 1);
  }
}
