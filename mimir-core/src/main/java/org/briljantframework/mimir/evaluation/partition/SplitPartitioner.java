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
 * The split partitioner simply partitions the input {@code DataFrame} (with {@code m rows}) and
 * {@code Vector} (of length {@code m}) into two parts (according to {@code testFraction} (in the
 * range {@code [0, 1]}).
 *
 * <p>
 * The training partition is of size {@code m*(1-testFraction)} and the validation partition of size
 * {@code m*testFraction}
 *
 * @author Isak Karlsson
 */
public class SplitPartitioner implements Partitioner {

  private final double testFraction;

  public SplitPartitioner(double testFraction) {
    Check.inRange(testFraction, 0, 1);
    this.testFraction = testFraction;
  }

  @Override
  public Collection<Partition> partition(DataFrame x, Vector y) {
    return new AbstractCollection<Partition>() {
      @Override
      public Iterator<Partition> iterator() {
        return new SplitIterator(x, y, testFraction);
      }

      @Override
      public int size() {
        return 1;
      }
    };
  }

  @Override
  public String toString() {
    return "SplitPartitioner{" + "testFraction=" + testFraction + '}';
  }

}
