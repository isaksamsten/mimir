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
package org.briljantframework.mimir.distance;

import java.util.Objects;

import org.briljantframework.data.vector.Vector;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class HammingDistance implements Distance<Vector> {

  @Override
  public double compute(double a, double b) {
    return a == b ? 0 : 1;
  }

  @Override
  public double compute(Vector a, Vector b) {
    int size = Math.min(a.size(), b.size());
    double distance = 0;
    for (int i = 0; i < size; i++) {
      Object av = a.loc().get(Object.class, i);
      Object bv = b.loc().get(Object.class, i);
      distance += Objects.equals(av, bv) ? 0 : 1;
    }

    return distance / size;
  }

  @Override
  public double max() {
    return Double.POSITIVE_INFINITY;
  }

  @Override
  public double min() {
    return Double.NEGATIVE_INFINITY;
  }
}
