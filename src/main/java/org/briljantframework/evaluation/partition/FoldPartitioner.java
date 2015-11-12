/*
 * The MIT License (MIT)
 * 
 * Copyright (c) 2015 Isak Karlsson
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

package org.briljantframework.evaluation.partition;

import java.util.AbstractCollection;
import java.util.Collection;
import java.util.Iterator;

import org.briljantframework.Check;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;

/**
 * Creates a k-fold partitioner
 * <p>
 *
 * @author Isak Karlsson
 */
public class FoldPartitioner implements Partitioner {

  private final int folds;

  public FoldPartitioner(int folds) {
    this.folds = folds;
  }

  @Override
  public Collection<Partition> partition(DataFrame x, Vector y) {
    Check.size(x.rows(), y.size());
    return new AbstractCollection<Partition>() {
      @Override
      public Iterator<Partition> iterator() {
        return new FoldIterator(x, y, folds);
      }

      @Override
      public int size() {
        return folds;
      }
    };
  }

  @Override
  public String toString() {
    return "FoldPartitioner{" + "folds=" + folds + '}';
  }
}
