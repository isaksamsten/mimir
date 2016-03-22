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
import org.briljantframework.data.vector.Vector;
import org.briljantframework.mimir.data.Input;

/**
 * @author Isak Karlsson
 */
public class MeanImputer<T extends DoubleSequence> implements Transformation<T, T> {

  @Override
  public Transformer<T, T> fit(Input<? extends T> frame) {
    Vector.Builder builder = Vector.Builder.of(Double.class);
    // for (Object key : frame.getColumnIndex().keySet()) {
    // builder.set(key, Vectors.mean(frame.get(key)));
    // }
    Vector means = builder.build();
    return new MeanImputeTransformer<>(means);
  }

  @Override
  public String toString() {
    return "MeanImputer{}";
  }

  private static class MeanImputeTransformer<T extends DoubleSequence>
      implements Transformer<T, T> {

    private final Vector means;

    public MeanImputeTransformer(Vector means) {
      this.means = means;
    }

    // @Override
    // public DataFrame transform(DataFrame x) {
    // Check.dimension(x.columns(), means.size());
    // DataFrame.Builder df = x.newBuilder();
    // for (Object colKey : x.getColumnIndex().keySet()) {
    // Vector column = x.get(colKey);
    // for (Object rowKey : x.getIndex().keySet()) {
    // if (column.isNA(rowKey)) {
    // df.set(rowKey, colKey, means, colKey);
    // } else {
    // df.set(rowKey, colKey, column, rowKey);
    // }
    // }
    // }
    // return df.build();
    // }

    @Override
    public String toString() {
      return "MeanImputTransformer{" + "means=" + means + '}';
    }

    @Override
    public Input<T> transform(Input<? extends T> x) {
      return null;
    }

    @Override
    public T transform(T x) {
      return null;
    }
  }
}
