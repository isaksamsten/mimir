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

import java.util.Iterator;
import java.util.Random;

import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.timeseries.data.TimeSeries;

public class RandomShapeletIterable implements Iterable<ShapeletWithIndex> {
  private final int noShapelets;
  private final Input<TimeSeries> timeSeries;
  private final double lowFrac, uppFrac;
  private final Random random;

  public RandomShapeletIterable(int noShapelets, Input<TimeSeries> timeSeries, double lowFrac,
      double uppFrac, Random random) {
    this.noShapelets = noShapelets;
    this.timeSeries = timeSeries;
    this.lowFrac = lowFrac;
    this.uppFrac = uppFrac;
    this.random = random;
  }

  @Override
  public Iterator<ShapeletWithIndex> iterator() {
    return new Iterator<ShapeletWithIndex>() {
      private int i = 0;

      @Override
      public boolean hasNext() {
        return i < noShapelets;
      }

      @Override
      public ShapeletWithIndex next() {
        i++;
        int index = random.nextInt(timeSeries.size());
        TimeSeries uts = timeSeries.get(index);
        int timeSeriesLength = uts.size();
        int lower = (int) Math.round(timeSeriesLength * lowFrac);
        int upper = (int) Math.round(timeSeriesLength * uppFrac);
        if (lower < 2) {
          lower = 2;
        }

        if (Math.addExact(upper, lower) > timeSeriesLength) {
          upper = timeSeriesLength - lower;
        }
        if (lower == upper) {
          upper -= 2;
        }
        if (upper < 1) {
          return null;
        }

        int length = random.nextInt(upper) + lower;
        int start = random.nextInt(timeSeriesLength - length);
        return new ShapeletWithIndex(index, new IndexSortedNormalizedShapelet(start, length, uts));
      }
    };
  }
}
