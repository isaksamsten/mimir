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
package org.briljantframework.mimir.timeseries.data;

import java.util.Arrays;
import java.util.Collection;

import org.apache.commons.math3.stat.descriptive.StatisticalSummary;
import org.briljantframework.DoubleVector;
import org.briljantframework.data.Collectors;
import org.briljantframework.data.series.Convert;
import org.briljantframework.data.series.Series;

/**
 * A simple double sequence
 *
 * @author Isak Karlsson
 */
public class TimeSeries implements DoubleVector {

  private double[] buffer;

  public TimeSeries(double[] x) {
    this.buffer = x;
    int ec = 0;
  }

  public static TimeSeries copyOf(Collection<?> data) {
    if (data instanceof Series) {
      return copyOf((Series) data);
    }
    double[] buffer = new double[data.size()];
    int i = 0;
    for (Object o : data) {
      buffer[i++] = Convert.to(Double.class, o);
    }
    return new TimeSeries(buffer);
  }

  public static TimeSeries copyOf(Series data) {
    double[] buffer = new double[data.size()];
    for (int i = 0; i < data.size(); i++) {
      buffer[i] = data.values().getDouble(i);
    }
    return new TimeSeries(buffer);
  }

  public static TimeSeries of(double... data) {
    return new TimeSeries(data);
  }

  public static TimeSeries normalizedCopyOf(Series data) {
    StatisticalSummary summary = data.collect(Double.class, Collectors.statisticalSummary());
    double[] x = new double[data.size()];
    for (int i = 0; i < data.size(); i++) {
      x[i] = (data.values().getDouble(i) - summary.getMean()) / summary.getStandardDeviation();
    }
    double sumDiff = 0;
    for (int i = 1; i < x.length; i++) {
      sumDiff += x[i] - x[i - 1];
    }

    sumDiff /= x.length - 1;

    boolean[] extremes = new boolean[x.length];
    for (int i = 1; i < x.length - 1; i++) {
      if ((x[i] > x[i - 1] && x[i] > x[i + 1]) || (x[i] < x[i - 1] && x[i] < x[i + 1])) {
        extremes[i] = true;
      } else if ((x[i + 1] - 2 * x[i] + x[i - 1]) > sumDiff) {
        extremes[i] = true;
      }
    }

    return new TimeSeries(x);
  }


  @Override
  public int size() {
    return buffer.length;
  }

  @Override
  public double getDouble(int index) {
    return buffer[index];
  }

  @Override
  public String toString() {
    return Arrays.toString(buffer);
  }
}
