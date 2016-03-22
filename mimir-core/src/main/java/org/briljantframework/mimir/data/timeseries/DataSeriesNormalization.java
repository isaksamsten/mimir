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

import org.briljantframework.DoubleSequence;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.transform.Transformer;

/**
 * <p>
 * Normalizes the rows of a {@link DataFrame} as opposed to the columns.
 * </p>
 *
 * <p>
 * This implementation performs a z-normalization of each {@code row} in a data frame.
 * Z-normalization ensures that all rows has approximately zero mean and unit variance.
 * </p>
 *
 * @author Isak Karlsson
 */
public class DataSeriesNormalization<T extends DoubleSequence> implements Transformer<T, T> {

  /**
   * <p>
   * Performs the transformation.
   * </p>
   *
   * <ul>
   * <li>Requires that
   * {@link org.briljantframework.data.index.VectorLocationGetter#getAsDouble(int)} returns a valid
   * double</li>
   * <li>Cannot handle {@code NA} values</li>
   * <li>Cannot handle {@link Double#NaN}</li>
   * </ul>
   *
   * @param x data frame to transform
   * @return a new data frame with normalized rows
   */
  @Override
  public Input<T> transform(Input<? extends T> x) {
    // DataFrame.Builder builder = x.newCopyBuilder();
    // for (int i = 0; i < x.rows(); i++) {
    // Vector row = x.loc().getRecord(i);
    // double mean = Vectors.mean(row);
    // double sigma = Vectors.std(row, mean);
    // for (int j = 0; j < row.size(); j++) {
    // double value = row.loc().getAsDouble(j);
    // builder.loc().set(i, j, (value - mean) / sigma);
    // }
    // }
    //
    // return builder.build();
    return null;
  }

  @Override
  public T transform(T x) {
    return null;
  }
}
