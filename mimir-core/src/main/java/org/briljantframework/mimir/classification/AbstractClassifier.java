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

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.IntStream;

import org.briljantframework.array.Array;
import org.briljantframework.mimir.data.Input;

/**
 * Provides sane defaults for a predictor. Sub-classes only have to implement the
 * {@link #predict(Object)} method to have a sensible default predictor.
 * 
 * <p/>
 * Predictors that produces probability estimates should make sure implement
 * {@link ProbabilityEstimator}.
 *
 * @author Isak Karlsson
 */
public abstract class AbstractClassifier<In, Out> implements Classifier<In, Out> {

  private final Array<Out> classes;

  protected AbstractClassifier(Array<Out> classes) {
    this.classes = Objects.requireNonNull(classes);
  }

  @Override
  public final Array<Out> getClasses() {
    return classes;
  }

  @Override
  @SuppressWarnings("unchecked")
  public List<Out> predict(Input<In> x) {
    // This guarantees that the order of predictions is the same as the input
    Object[] labels = new Object[x.size()];
    IntStream.range(0, x.size()).parallel().forEach(i -> labels[i] = predict(x.get(i)));
    return Arrays.asList((Out[]) labels);
  }

}
