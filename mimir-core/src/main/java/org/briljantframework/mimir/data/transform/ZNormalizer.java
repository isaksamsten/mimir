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
 * Z normalization is also known as "Normalization to Zero Mean and Unit of Energy" first mentioned
 * in Goldin & Kanellakis. It ensures that all elements of the input vector are transformed into the
 * output vector whose mean is approximately 0 while the standard deviation are in a range close to
 * 1.
 *
 * @author Isak Karlsson
 */
public class ZNormalizer<T extends DoubleSequence> implements Transformation<T, T> {

  @Override
  public Transformer<T, T> fit(Input<? extends T> df) {
    Vector.Builder meanBuilder = Vector.Builder.of(Double.class);
    Vector.Builder stdBuilder = Vector.Builder.of(Double.class);
    // for (Object columnKey : df) {
    // StatisticalSummary stats = Vectors.statisticalSummary(df.get(columnKey));
    // if (stats.getN() <= 0 || Is.NA(stats.getMean()) || Is.NA(stats.getStandardDeviation())) {
    // throw new IllegalArgumentException("Illegal value for column " + columnKey);
    // }
    // meanBuilder.set(columnKey, stats.getMean());
    // stdBuilder.set(columnKey, stats.getStandardDeviation());
    // }
    Vector mean = meanBuilder.build();
    Vector sigma = stdBuilder.build();

    return new ZNormalizerTransformer<>(mean, sigma);
  }

  @Override
  public String toString() {
    return "ZNormalizer";
  }

  private static class ZNormalizerTransformer<T extends DoubleSequence>
      implements Transformer<T, T> {

    private final Vector mean;
    private final Vector sigma;

    ZNormalizerTransformer(Vector mean, Vector sigma) {
      this.mean = mean;
      this.sigma = sigma;
    }

    @Override
    public Input<T> transform(Input<? extends T> x) {
      // Check.argument(mean.getIndex().equals(x.getColumnIndex()), "Columns must match.");
      // DataFrame.Builder builder = x.newBuilder();
      // for (Object columnKey : x) {
      // Vector column = x.get(columnKey);
      // double m = mean.getAsDouble(columnKey);
      // double std = sigma.getAsDouble(columnKey);
      // Vector.Builder normalized = column.newBuilder(column.size());
      // for (int i = 0, size = column.size(); i < size; i++) {
      // double v = column.loc().getAsDouble(i);
      // if (Is.NA(v)) {
      // normalized.addNA();
      // } else if (std == 0) {
      // normalized.add(0);
      // } else {
      // normalized.add((v - m) / std);
      // }
      // }
      // builder.set(columnKey, normalized);
      // }
      // return builder.setIndex(x.getIndex()).build();
      throw new UnsupportedOperationException();
    }

    @Override
    public T transform(T x) {
      throw new UnsupportedOperationException();
    }

    @Override
    public String toString() {
      return "ZNormalizerTransformer{" + "mean=" + mean + ", sigma=" + sigma + '}';
    }
  }
}
