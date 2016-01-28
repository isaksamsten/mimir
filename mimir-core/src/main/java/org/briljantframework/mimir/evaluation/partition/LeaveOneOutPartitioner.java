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
package org.briljantframework.mimir.evaluation.partition;

import java.util.AbstractCollection;
import java.util.Collection;
import java.util.Iterator;

import org.briljantframework.Check;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;

/**
 * The leave-one-out partitioner can be used to implement Leave-one-out cross-validation, a commonly
 * employed strategy for evaluating small and expensive to gather datasets.
 *
 * <p/>
 * The {@linkplain DataFrame} (with {@code m} rows) and {@linkplain Vector} (of length {@code m})
 * are partitioned into {@code m} partitions. At each iteration {@code m-1} data points are returned
 * as the training set and {@code 1} data point as the validation set. All data points are used as
 * validation points exactly once.
 *
 * @author Isak Karlsson
 * @see FoldPartitioner
 */
public class LeaveOneOutPartitioner implements Partitioner {

  @Override
  public Collection<Partition> partition(DataFrame x, Vector y) {
    Check.dimension(x.rows(), y.size());
    return new AbstractCollection<Partition>() {
      @Override
      public Iterator<Partition> iterator() {
        return new FoldIterator(x, y, x.rows());
      }

      @Override
      public int size() {
        return x.rows();
      }
    };
  }

  @Override
  public String toString() {
    return "LeaveOneOutPartitioner";
  }
}
