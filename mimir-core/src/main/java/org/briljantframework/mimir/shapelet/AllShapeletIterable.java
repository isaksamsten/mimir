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

import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.timeseries.data.TimeSeries;

import java.util.Iterator;

public class AllShapeletIterable implements Iterable<ShapeletWithIndex> {

  private final Input<TimeSeries> timeSeries;
  private final int minSize, maxSize;

  public AllShapeletIterable(Input<TimeSeries> timeSeries, int minSize, int maxSize) {
    this.timeSeries = timeSeries;
    this.minSize = minSize;
    this.maxSize = maxSize;
  }

  @Override
  public Iterator<ShapeletWithIndex> iterator() {
    return new Iterator<ShapeletWithIndex>() {

      int i = 1;
      Iterator<Shapelet> it = new ShapeletIterable(timeSeries.get(0), minSize, maxSize);

      @Override
      public boolean hasNext() {
        return i < timeSeries.size() || it.hasNext();
      }

      @Override
      public ShapeletWithIndex next() {
        if (!it.hasNext()) {
          it = new ShapeletIterable(timeSeries.get(i++), minSize, maxSize);
        }
        return new ShapeletWithIndex(i, it.next());
      }
    };
  }

  private static class ShapeletIterable implements Iterator<Shapelet> {

    private final TimeSeries timeSeries;
    private final int max, min;
    private int j = 0, k;


    public ShapeletIterable(TimeSeries timeSeries, int min, int max) {
      this.timeSeries = timeSeries;
      this.max = max < 0 ? timeSeries.size() : max;
      this.min = min;
      this.k = this.min;

    }

    @Override
    public boolean hasNext() {
      return j < timeSeries.size() && k <= max - j;
    }

    @Override
    public Shapelet next() {
      IndexSortedNormalizedShapelet indexSortedNormalizedShapelet =
          new IndexSortedNormalizedShapelet(j, k++, timeSeries);
      if (k == (max - j) + 1) {
        j++;
        k = min;
      }
      return indexSortedNormalizedShapelet;
    }
  }
}
