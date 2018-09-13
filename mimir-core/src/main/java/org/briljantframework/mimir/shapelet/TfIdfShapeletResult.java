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

import java.util.List;

import org.briljantframework.array.DoubleArray;

/**
 * Created by isak on 2017-06-19.
 */
public class TfIdfShapeletResult {
  private final int start;
  private final int length;
  private final int timeSeries;
  private final DoubleArray scores;
  private final List<ShapeletMatch> matches;
  private final double impurity, threshold;
  private final int shapeletId;

  TfIdfShapeletResult(int shapeletId, int start, int length, int timeSeries, double impurity, double threshold,
      DoubleArray scores, List<ShapeletMatch> matches) {
    this.shapeletId = shapeletId;
    this.start = start;
    this.length = length;
    this.timeSeries = timeSeries;
    this.impurity = impurity;
    this.scores = scores;
    this.matches = matches;
    this.threshold = threshold;
  }

  public int getShapeletId() {
    return shapeletId;
  }

  public List<ShapeletMatch> getMatches() {
    return matches;
  }

  public int getStart() {
    return start;
  }

  public int getLength() {
    return length;
  }

  public int getTimeSeries() {
    return timeSeries;
  }

  public DoubleArray getScores() {
    return scores;
  }

  @Override
  public String toString() {
    return scores.toString();
  }

  public double getImpurity() {
    return impurity;
  }

  public double getThreshold() {
    return threshold;
  }
}
