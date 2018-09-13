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

import java.util.Arrays;

import org.briljantframework.array.DoubleArray;

/**
 * Created by isak on 2017-03-28.
 */
public class ShapeletScore implements Comparable<ShapeletScore> {

  private final double[] shapelet;
  private final Shapelet first;
  private double score;
  private int no = 1;

  public ShapeletScore(Shapelet shapelet, double score) {
    this.first = shapelet;
    this.shapelet = new double[shapelet.size()];
    for (int i = 0; i < shapelet.size(); i++) {
      this.shapelet[i] = shapelet.getDouble(i);
    }

    this.score = score;
  }

  public Shapelet getPrototypeShapelet() {
    return first;
  }

  public double getScore() {
    return score / no;
  }

  public void incrementScore(Shapelet s, double score) {
    this.score += score;
    this.no += 1;
    for (int i = 0; i < s.size(); i++) {
      shapelet[i] += s.getDouble(i);
    }
  }

  @Override
  public int compareTo(ShapeletScore o) {
    return Double.compare(getScore(), o.getScore());
  }

  @Override
  public String toString() {
    double[] s = new double[shapelet.length];
    for (int i = 0; i < shapelet.length; i++) {
      s[i] = shapelet[i] / no;
    }
    return getScore()+ ": " + Arrays.toString(s);
  }
}
