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

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.briljantframework.Check;
import org.briljantframework.data.series.Series;
import org.briljantframework.mimir.data.timeseries.SymbolicAggregator;

/**
 * @author Isak Karlsson
 */
public class SaxDistance implements Distance<Series> {

  private final Map<String, Map<String, Double>> lookup;
  private final double n;

  public SaxDistance(double n, Map<String, Map<String, Double>> lookup) {
    this.lookup = lookup;
    this.n = n;
  }

  public SaxDistance(double n, String... alphabet) {
    this(n, Arrays.asList(alphabet));
  }

  public SaxDistance(double n, List<String> alphabet) {
    this(n, SymbolicAggregator.newLookupTable(alphabet));
  }

  @Override
  public double compute(Series a, Series b) {
    Check.dimension(a.size(), b.size());

    double w = a.size();
    double sum = 0;

    for (int i = 0; i < w; i++) {
      String av = a.loc().get(String.class, i);
      String bv = b.loc().get(String.class, i);
      double value = lookup.get(av).get(bv);
      sum += value * value;
    }
    return Math.sqrt(n / w) * Math.sqrt(sum);
  }

}
