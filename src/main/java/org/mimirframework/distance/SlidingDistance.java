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

package org.mimirframework.distance;

import org.briljantframework.data.vector.Vector;
import org.mimirframework.shapelet.Shapelet;

/**
 * Created by isak on 24/03/15.
 */
public class SlidingDistance implements Distance {

  private final Distance distanceMeasure;

  public SlidingDistance(Distance instance) {
    this.distanceMeasure = instance;
  }

  @Override
  public double compute(double a, double b) {
    return 0;
  }

  @Override
  public double compute(Vector a, Vector b) {
    double minDistance = Double.POSITIVE_INFINITY;

    // Assumed to be normalized!
    Vector candidate = a.size() < b.size() ? a : b;
    Vector vector = a.size() >= b.size() ? a : b;
    for (int i = 0; i <= vector.size() - candidate.size(); i++) {
      Shapelet subShapelet = new Shapelet(i, candidate.size(), vector);
      double sumDistance = distanceMeasure.compute(candidate, subShapelet);
      if (sumDistance < minDistance) {
        minDistance = sumDistance;
      }
    }
    return Math.sqrt(minDistance / candidate.size());
  }

  @Override
  public double max() {
    return 0;
  }

  @Override
  public double min() {
    return 0;
  }
}
