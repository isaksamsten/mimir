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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.briljantframework.Check;
import org.briljantframework.DoubleSequence;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.mimir.data.Input;


/**
 * Utilities for creating aggregated data frames.
 * 
 * @author Isak Karlsson
 */
public final class Approximations {

  private static final List<String> alphabet =
      Collections.unmodifiableList(Arrays.asList("a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
          "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "x", "y", "z"));

  private Approximations() {}

  /**
   * Performs Piecewise Aggregate Approximation, reducing each row of {@code in} to length
   * {@code size}
   *
   * @param in the input data frame
   * @param size the resulting time series size
   * @return a new data frame with {@link org.briljantframework.data.dataframe.DataFrame#columns()}
   *         equals to {@code size}
   */
  public static <T extends DoubleSequence> Input<T> paa(Input<? extends T> in, int size) {
    return Approximations.<T>paa(size).transform(in);
  }

  /**
   * Returns a {@link org.briljantframework.mimir.data.transform.Transformer} that reduces the
   * length of each row-vector to {@code size}.
   *
   * @param size the resulting time series size
   * @return a transformation
   */
  public static <T extends DoubleSequence> AggregateApproximation<T> paa(int size) {
    return new AggregateApproximation<>(size);
  }

  public static DataFrame sax(DataFrame in, int size) {
    return sax(in, size, getAlphabet(size));
  }

  public static List<String> getAlphabet(int size) {
    Check.argument(size < alphabet.size(), "Alphabet size to large.");
    return alphabet.subList(0, size);
  }

  public static DataFrame sax(DataFrame in, int size, List<String> alphabet) {
    // Check.argument(alphabet.size() > 1, "Alphabet size must be larger than 1.");
    // return PipelineTransformer.of(paa(size), sax(alphabet)).transform(in);
    throw new UnsupportedOperationException();
  }

  /**
   * Returns a {@link org.briljantframework.mimir.data.transform.Transformer} that reduces the size
   * and transforms each row in the input data frame to
   */
  public static AggregateApproximation sax(List<String> alphabet) {
    return new AggregateApproximation(new SymbolicAggregator(alphabet));
  }

  public static DataFrame sax(DataFrame in, int size, String... alphabet) {
    return sax(in, size, Arrays.asList(alphabet));
  }

}
