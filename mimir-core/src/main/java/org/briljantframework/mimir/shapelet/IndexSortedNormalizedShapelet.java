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
package org.briljantframework.mimir.shapelet;

import java.util.function.IntBinaryOperator;

import org.briljantframework.DoubleSequence;
import org.briljantframework.util.sort.QuickSort;

/**
 * @author Isak Karlsson
 */
public class IndexSortedNormalizedShapelet extends NormalizedShapelet {
  protected final int[] order;

  public IndexSortedNormalizedShapelet(int start, int length, DoubleSequence vector) {
    super(start, length, vector);
    if (vector instanceof IndexSortedNormalizedShapelet) {
      this.order = ((IndexSortedNormalizedShapelet) vector).getSortOrder();
    } else {
      this.order = indexSort(size(),
          (i, j) -> Double.compare(Math.abs(getDouble(j)), Math.abs(getDouble(i))));
    }
  }

  private int[] indexSort(int size, IntBinaryOperator operator) {
    int[] indicies = new int[size];
    for (int i = 0; i < indicies.length; i++) {
      indicies[i] = i;
    }
    QuickSort.quickSort(0, indicies.length, (a, b) -> operator.applyAsInt(indicies[a], indicies[b]),
        (a, b) -> {
          int tmp = indicies[a];
          indicies[a] = indicies[b];
          indicies[b] = tmp;
        });
    return indicies;
  }

  public int[] getSortOrder() {
    return order;
  }
}
