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
package org.briljantframework.mimir.data.transform;

import org.briljantframework.DoubleSequence;
import org.briljantframework.data.series.Series;
import org.briljantframework.mimir.data.Input;

/**
 * Class to fit a min max normalizer to a data frame. Calculate, for each column {@code j}, the min
 * <i>min</i><minus>j</minus> and max <i>max</i><minus>j</minus>. Then, for each value x
 * <minus>i,j</minus> is given by (x<minus>i,j</minus>-min<minus>j</minus>)/(max<minus>j</minus> -
 * min<minus>j</minus>). This normalizes the data frame in the range {@code [0, 1]} (under the
 * assumption that min and max are representative for the transformed dataframe).
 *
 * @author Isak Karlsson
 */
public class MinMaxNormalizer<T extends DoubleSequence> implements Transformation<T, T> {

  @Override
  public Transformer<T, T> fit(Input<? extends T> x) {
    Series.Builder min = Series.Builder.of(Double.class);
    Series.Builder max = Series.Builder.of(Double.class);
    // for (Object columnKey : df) {
    // TODO: 11/14/15 Only consider numerical vectors
    // StatisticalSummary summary = df.get(columnKey).statisticalSummary();
    // min.set(columnKey, summary.getMin());
    // max.set(columnKey, summary.getMax());
    // }

    return new MinMaxNormalizeTransformer<>(max.build(), min.build());
  }

  @Override
  public String toString() {
    return "MinMaxNormalizer";
  }

  private static class MinMaxNormalizeTransformer<T extends DoubleSequence>
      implements Transformer<T, T> {

    private final Series max;
    private final Series min;

    public MinMaxNormalizeTransformer(Series max, Series min) {
      this.max = max;
      this.min = min;
    }

    @Override
    public Input<T> transform(Input<? extends T> x) {
      // Check.argument(max.getIndex().equals(x.getColumnIndex()), "Index does not match");
      // DataFrame.Builder builder = x.newBuilder();
      // for (Object columnKey : x) {
      // TODO: 11/14/15 Only consider numerical vectors
      // double min = this.min.getAsDouble(columnKey);
      // double max = this.max.getAsDouble(columnKey);
      // Series.Builder normalized = Series.Builder.of(Double.class);
      // Series column = x.get(columnKey);
      // for (int i = 0, size = column.size(); i < size; i++) {
      // double v = column.loc().getAsDouble(i);
      // if (Is.NA(v)) {
      // normalized.addNA();
      // } else {
      // normalized.add((column.loc().getAsDouble(i) - min) / (max - min));
      // }
      // }
      // builder.set(columnKey, normalized);
      // }
      return null;// builder.setIndex(x.getIndex()).build();
    }

    @Override
    public T transform(T x) {
      return null;
    }

    @Override
    public String toString() {
      return "MinMaxNormalizeTransformer{" + "max=" + max + ", min=" + min + '}';
    }
  }
}
