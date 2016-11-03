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
package org.briljantframework.mimir.classification;

import java.util.stream.IntStream;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.data.Input;

/**
 * Created by isak on 2016-10-25.
 */
public interface ProbabilityEstimator<In, Out> extends Classifier<In, Out> {

  @Override
  default Out predict(In input) {
    return getClasses().get(Arrays.argmax(estimate(input)));
  }

  /**
   * Estimates the posterior probabilities for all objects in the given input
   *
   * <p>
   * Each column corresponds to the probability of a particular class (the j:th column correspond to
   * the j:th element in {@linkplain #getClasses()}) and each row corresponds to a particular record
   * in the supplied data frame.
   *
   * @param x the data frame of records to estimate the posterior probabilities for
   * @return a matrix with probability estimates; shape = {@code [x.rows(), getClasses().size()]}.
   */
  default DoubleArray estimate(Input<? extends In> x) {
    DoubleArray estimations = DoubleArray.zeros(x.size(), getClasses().size());
    IntStream.range(0, x.size()).parallel().forEach(i -> estimations.setRow(i, estimate(x.get(i))));
    return estimations;
  }

  /**
   * Estimates the posterior probability for the supplied input.
   *
   * <p>
   * The i:th element in the returned array correspond to the probability of the i:th class in
   * {@linkplain #getClasses()}
   *
   * @param input the input to estimate the posterior probability for
   * @return an array with probability estimates; shape = {@code [1, this.getClasses().size()]}.
   */
  DoubleArray estimate(In input);
}
