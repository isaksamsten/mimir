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

import org.briljantframework.data.vector.Vector;

/**
 * A z-normalized sub sequence view of another MatrixLike
 * <p>
 * Created by Isak Karlsson on 28/09/14.
 */
public class NormalizedShapelet extends Shapelet {

  private final double sigma;
  private final double mean;

  public NormalizedShapelet(int start, int length, Vector vector) {
    super(start, length, vector);
    if (vector instanceof NormalizedShapelet) {
      this.sigma = ((NormalizedShapelet) vector).sigma;
      this.mean = ((NormalizedShapelet) vector).mean;
    } else {
      double ex = 0;
      double ex2 = 0;
      int size = start + length;
      for (int i = start; i < size; i++) {
        double v = vector.loc().getAsDouble(i);
        ex += v;
        ex2 += v * v;
      }
      this.mean = ex / length;
      if (length == 1) {
        this.sigma = 0;
      } else {
        this.sigma = Math.sqrt(ex2 / length - mean * mean);
      }
    }
  }

  /**
   * Create normalized shapelet.
   *
   * @param start the start
   * @param length the length
   * @param vectorLike the vector like
   * @return the normalized shapelet
   */
  public static NormalizedShapelet create(int start, int length, Vector vectorLike) {
    return new NormalizedShapelet(start, length, vectorLike);
  }

  @Override
  public double getAsDoubleAt(int i) {
    if (sigma == 0) {
      return 0;
    } else {
      return (super.getAsDoubleAt(i) - mean) / sigma;
    }
  }
}
