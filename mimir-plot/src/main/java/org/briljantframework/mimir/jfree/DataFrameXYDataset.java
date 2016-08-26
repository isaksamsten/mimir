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

import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.jfree.data.xy.AbstractXYDataset;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class DataFrameXYDataset extends AbstractXYDataset {

  private final DataFrame dataFrame;

  public DataFrameXYDataset(DataFrame dataFrame) {
    this.dataFrame = dataFrame;
  }

  @Override
  public int getSeriesCount() {
    return dataFrame.rows();
  }

  @Override
  public Comparable getSeriesKey(int series) {
    return (Comparable) dataFrame.getColumnIndex().get(series);
  }

  @Override
  public int getItemCount(int series) {
    return dataFrame.columns();
  }

  @Override
  public Number getX(int series, int item) {
    return item;
  }

  @Override
  public Number getY(int series, int item) {
    Object r = dataFrame.getIndex().get(item);
    Object c = dataFrame.getColumnIndex().get(series);
    Number value = dataFrame.get(Number.class, r, c);
    if (Is.NA(value)) {
      return null;
    }
    return value;
  }

}
