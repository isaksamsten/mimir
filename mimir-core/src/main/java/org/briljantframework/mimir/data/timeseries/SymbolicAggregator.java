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

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.series.Series;
import org.briljantframework.data.series.Type;
import org.briljantframework.data.series.Types;

/**
 * Symbolic aggregation (when normalized) is a representation method for data series that allow for
 * lower bounding the distance calculation. It works by transforming the normalized data series into
 * a series of discrete symbols, called words, by mapping each data point using the probability
 * associated with a particular value.
 * <p/>
 *
 * For example, given the alphabet {@code [a, b, c, d]} the following thresholds can be computed
 * from the normal distribution and equally sized regions {@code [-0.674,0.000,0.674]}. Then, if a
 * value is {@code <= -0.674} assign {@code a}, if the value is {@code >= 0.674} assign {@code d},
 * if value is {@code -0.674 < value < 0} assign b and if {@code 0 < value < 0.674} assign c. That
 * is, {@code [-1, 1, 0.3, -0.2]} becomes {@code [a, d, c, b]}.
 * <p/>
 *
 * <p>
 * The {@link SymbolicAggregator} is often coupled with the {@link MeanAggregator}. For example,
 *
 * <pre>
 * Transformer sax = PipelineTransformer.of(new AggregateApproximation(new MeanAggreagator(5)),
 *     new AggregateApproximation(new SymbolicAggregator(&quot;a&quot;, &quot;b&quot;, &quot;c&quot;, &quot;d&quot;)));
 * sax.transform(x);
 * 
 * // Alternatively
 * Approximations.sax(x, 5, &quot;a&quot;, &quot;b&quot;, &quot;c&quot;, &quot;d&quot;);
 * </pre>
 *
 * @author Isak Karlsson
 */
public class SymbolicAggregator implements Aggregator {

  private final Series alphabet;
  private final DoubleArray thresholds;

  /**
   * Constructs a symbolic aggregator using {@code alphabet}
   *
   * @param alphabet the alphabet
   */
  public SymbolicAggregator(List<String> alphabet) {
    this(Series.of(alphabet));
  }

  /**
   * Constructs a symbolic aggregator using {@code alphabet}
   *
   * @param alphabet the alphabet
   */
  public SymbolicAggregator(Series alphabet) {
    this.alphabet = alphabet;
    thresholds = calculateThresholds(alphabet);
  }

  /*
   * Compute the thresholds for the alphabet using the normal distribution. Given an alphabet A,
   * computes the thresholds as [ppf(1/|A|), ppf(2/|A|), ..., ppf((|A|-1)/|A|)].
   */
  private static DoubleArray calculateThresholds(Series alphabet) {
    double prob = 1.0 / alphabet.size();
    int length = alphabet.size() - 1;
    RealDistribution distribution = new NormalDistribution(0, 1);
    DoubleArray array = Arrays.linspace(prob, 1.0 - prob, length);
    array.map(distribution::inverseCumulativeProbability);
    return array;
  }

  /**
   * Constructs a symbolic aggregator using {@code alphabet}
   *
   * @param alphabet the alphabet
   */
  public SymbolicAggregator(String... alphabet) {
    this(Series.of(alphabet));
  }

  /**
   * Gets the lookup table, used for similarity calculations, as produced by {@code alphabet}
   *
   * @param alphabet the alphabet
   * @return the lookup table
   */
  public static Map<String, Map<String, Double>> newLookupTable(List<String> alphabet) {
    Series vector = Series.of(alphabet); // TODO: note of(...)
    return createLookupTable(vector, calculateThresholds(vector));
  }

  /*
   * Creates a lookup table for the similarity between two items in alphabet
   */
  private static Map<String, Map<String, Double>> createLookupTable(Series alphabet,
      DoubleArray thresholds) {
    Map<String, Map<String, Double>> tab = new HashMap<>();
    for (int r = 0; r < alphabet.size(); r++) {
      Map<String, Double> sub = new HashMap<>();
      tab.put(alphabet.loc().get(String.class, r), sub);
      for (int c = 0; c < alphabet.size(); c++) {
        if (Math.abs(r - c) <= 1) {
          sub.put(alphabet.loc().get(String.class, c), 0.0);
        } else {
          sub.put(alphabet.loc().get(String.class, c),
              thresholds.get(Math.max(r, c) - 1) - thresholds.get(Math.min(r, c)));
        }
      }
    }
    return tab;
  }

  /**
   * Gets the lookup table, used for computing the similarity, produced by the alphabet used for
   * this aggregator
   *
   * @return a lookup table
   */
  public Map<String, Map<String, Double>> getLookupTable() {
    return createLookupTable(alphabet, thresholds);
  }

  @Override
  public Type getAggregatedType() {
    return Types.from(String.class);
  }

  @Override
  public Series.Builder partialAggregate(Series in) {
    Series.Builder sax = Series.Builder.of(String.class);
    for (int j = 0; j < in.size(); j++) {
      double value = in.loc().getDouble(j);
      if (value <= thresholds.get(0)) {
        sax.loc().set(j, alphabet.loc().get(String.class, 0));
      } else if (value >= thresholds.get(thresholds.size() - 1)) {
        sax.loc().set(j, alphabet.loc().get(String.class, alphabet.size() - 1));
      } else {
        int index = 0;
        for (int k = 0; k < thresholds.size(); k++) {
          if (thresholds.get(k) <= value) {
            index = k;
          } else {
            break;
          }
        }
        sax.loc().set(j, alphabet.loc().get(String.class, index + 1));
      }
    }
    return sax;
  }
}
