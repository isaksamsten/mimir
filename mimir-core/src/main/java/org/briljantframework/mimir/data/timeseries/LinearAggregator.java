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


import org.briljantframework.Check;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.data.vector.Type;

/**
 * Linearly connects points.
 * 
 * @author Isak Karlsson
 */
public class LinearAggregator implements Aggregator {

  private final int targetSize;

  public LinearAggregator(int targetSize) {
    this.targetSize = targetSize;
  }

  @Override
  public Type getAggregatedType() {
    return Type.DOUBLE;
  }

  @Override
  public Vector.Builder partialAggregate(Vector in) {
    Check.argument(in.size() > targetSize, "Can't linearly oversample data series.");

    Vector.Builder builder = Vector.Builder.withCapacity(Double.class, targetSize);
    int bin = in.size() / targetSize;
    int pad = in.size() % targetSize;

    int currentIndex = 0;
    int toPad = 0;
    while (currentIndex < in.size()) {
      int inc = 0;

      // In some cases in.size() / targetSize result in a reminder,
      // distribute this reminder equally over all bins
      if (toPad++ < pad) {
        inc = 1;
      }
      int binInc = bin + inc;
      int start = currentIndex;
      int end = currentIndex + binInc - 1;
      double w = (double) 1 / 5;
      builder.add(lerp(in.loc().getAsDouble(start), in.loc().getAsDouble(end), w));
      currentIndex += binInc;
    }
    return builder;
  }

  private double lerp(double a, double b, double w) {
    return ((1 - w) * a) + (w * b);
  }
}
