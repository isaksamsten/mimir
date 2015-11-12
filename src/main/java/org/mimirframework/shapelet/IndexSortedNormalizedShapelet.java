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

package org.mimirframework.shapelet;

import org.briljantframework.data.vector.Vector;
import org.briljantframework.data.vector.Vectors;

/**
 * Created by Isak Karlsson on 27/10/14.
 */
public class IndexSortedNormalizedShapelet extends NormalizedShapelet {

  /**
   * The Order.
   */
  protected final int[] order;

  /**
   * Instantiates a new Index sorted normalized shapelet.
   *
   * @param start the start
   * @param length the length
   * @param vector the vector
   */
  public IndexSortedNormalizedShapelet(int start, int length, Vector vector) {
    super(start, length, vector);
    if (vector instanceof IndexSortedNormalizedShapelet) {
      this.order = ((IndexSortedNormalizedShapelet) vector).getSortOrder();
    } else {
      this.order =
          Vectors.indexSort(
              this,
              (i, j) -> Double.compare(Math.abs(loc().getAsDouble(j)),
                  Math.abs(loc().getAsDouble(i))));
    }
  }

  /**
   * Get order.
   *
   * @return the int [ ]
   */
  public int[] getSortOrder() {
    return order;
  }
}
