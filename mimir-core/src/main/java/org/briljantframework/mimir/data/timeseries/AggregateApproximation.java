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

import org.briljantframework.DoubleSequence;
//import org.briljantframework.data.dataseries.DataSeriesCollection;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.transform.InputTransformer;

/**
 * Transforms a DataFrame by, for each row, apply the injected {@link Aggregator} and reduce the
 * dimensionality of the record.
 * 
 * <p/>
 * For example, a common choice is the Piecewise Aggregate Approximation which divides each time
 * series into {@code n} bins. Each bin is averaged to produce a new data series of a specified
 * length. Since this i very common transformation,
 * {@link AggregateApproximation#AggregateApproximation(int)} produces an aggregate approximation of
 * length {@code size}.
 *
 * @author Isak Karlsson
 */
public class AggregateApproximation<T extends DoubleSequence> implements InputTransformer<T, T> {

  private final Aggregator aggregator;

  /**
   * Piecewise aggregate approximation (PAA) producing a new DataFrame with data series of length
   * {@code size}. Same as {@code new AggregateApproximation(new MeanAggregate(size))}.
   *
   * @param size the target size
   */
  public AggregateApproximation(int size) {
    this(new MeanAggregator(size));
  }

  /**
   * @param aggregator the aggregator
   */
  public AggregateApproximation(Aggregator aggregator) {
    this.aggregator = Objects.requireNonNull(aggregator, "Requires an aggregator.");
  }

  /**
   * Gets the aggregator
   *
   * @return the aggregator
   */
  public Aggregator getAggregator() {
    return aggregator;
  }

  @Override
  public Input<T> transform(Input<T> x) {
    // DataSeriesCollection.Builder builder =
    // new DataSeriesCollection.Builder(aggregator.getAggregatedType());
    // for (int i = 0; i < x.size(); i++) {
    // builder.addRecord(aggregator.partialAggregate(x.loc().getRecord(i)));
    // }
    // return builder.build();
    return null;
  }

  @Override
  public T transformElement(T x) {
    return null;
  }
}
