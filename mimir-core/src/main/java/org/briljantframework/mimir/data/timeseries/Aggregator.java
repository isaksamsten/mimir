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

import org.briljantframework.data.vector.Vector;
import org.briljantframework.data.vector.Type;

/**
 * Performs an aggregation of a particular data series, represented as a vector. For example, one
 * approach to aggregate {@code [1,2,3,4,5,6]}, from {@code 6} elements to, lets say, 3 elements, is
 * to divide the vector into {@code 3} equally sized bins: {@code [1,2], [3,4], [5,6]}. These bins
 * can then be aggregated by, e.g., averaging to produce the vector {@code [1.5, 3.5, 5.5]}.
 * <p/>
 * 
 * The {@link #partialAggregate(org.briljantframework.data.vector.Vector)}, takes an input vector
 * and produces a mutable {@link org.briljantframework.data.vector.Vector.Builder} for futher
 * transformation. The {@link #aggregate(org.briljantframework.data.vector.Vector)} produces a new,
 * aggregated, vector.
 * 
 * @author Isak Karlsson
 */
public interface Aggregator {

  /**
   * @return the vector type of the approximation
   */
  Type getAggregatedType();

  /**
   * Performs aggregation on {@code in}. Returns the resulting vector.
   * 
   * @param in the vector
   * @return an aggregate of {@code in}
   */
  default Vector aggregate(Vector in) {
    return partialAggregate(in).build();
  }

  /**
   * Perform aggregation on {@code in}. Return a mutable {@code Builder} for further processing.
   * 
   * @param in the vector
   * @return a mutable aggregate of {@code in}
   */
  Vector.Builder partialAggregate(Vector in);
}
