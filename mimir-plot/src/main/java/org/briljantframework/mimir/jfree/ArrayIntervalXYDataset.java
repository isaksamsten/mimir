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
package org.briljantframework.mimir.jfree;

import org.briljantframework.array.Array;
import org.jfree.data.xy.IntervalXYDataset;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ArrayIntervalXYDataset extends ArrayXYDataset implements IntervalXYDataset {

  private final Array<? extends Number> error;

  public ArrayIntervalXYDataset(Array<? extends Number> x, Array<? extends Number> y,
      Array<? extends Number> error) {
    super(x, y);
    this.error = error;
  }

  @Override
  public Number getStartX(int series, int item) {
    return getXValue(series, item);
  }

  @Override
  public double getStartXValue(int series, int item) {
    return getStartX(series, item).doubleValue();
  }

  @Override
  public Number getEndX(int series, int item) {
    return getXValue(series, item);
  }

  @Override
  public double getEndXValue(int series, int item) {
    return getEndX(series, item).doubleValue();
  }

  @Override
  public Number getStartY(int series, int item) {
    return getStartYValue(series, item);
  }

  @Override
  public double getStartYValue(int series, int item) {
    return getYValue(series, item) - error.get(item, series).doubleValue();
  }

  @Override
  public Number getEndY(int series, int item) {
    return getEndYValue(series, item);
  }

  @Override
  public double getEndYValue(int series, int item) {
    return getYValue(series, item) + error.get(item, series).doubleValue();
  }
}
