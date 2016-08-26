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

import org.briljantframework.data.series.Series;

/**
 * Manhattan distance, i.e sum of absolute difference
 * <p>
 * Created by Isak Karlsson on 01/09/14.
 */
public class ManhattanDistance implements Distance<Series> {

  private static Distance instance = new ManhattanDistance();

  private ManhattanDistance() {}

  public static Distance getInstance() {
    return instance;
  }

  @Override
  public double compute(Series a, Series b) {
    int size = Math.min(a.size(), b.size());
    double distance = 0.0;
    for (int i = 0; i < size; i++) {
      distance += Math.abs(a.loc().getDouble(i) - b.loc().getDouble(i));
    }
    return distance;
  }

  @Override
  public String toString() {
    return "ManhattanDistance";
  }
}
