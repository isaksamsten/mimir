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

import java.util.*;
import java.util.stream.IntStream;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Output;
import org.briljantframework.mimir.data.Outputs;
import org.briljantframework.mimir.supervised.Characteristic;

/**
 * Provides sane defaults for a predictor. Sub-classes only have to implement the
 * {@link #estimate(In)} method to have a sensible default predictor.
 * 
 * <p/>
 * For a classifier unable to output probability estimates the {@linkplain #estimate(In)} should
 * return an array where the predicted class has probability {@code 1} and all other classes
 * probability {@code 0}. To improve performance, implementors of such predictors should consider
 * overriding the default {@linkplain #predict(In)} method.
 * 
 * <p/>
 * Predictors that produces probability estimates should make sure to include the
 * {@link ClassifierCharacteristic#ESTIMATOR ESTIMATOR} characteristics in the {@link EnumSet}
 * returned by {@link #getCharacteristics()}
 *
 * @author Isak Karlsson
 */
public abstract class AbstractClassifier<In> implements Classifier<In> {

  private final List<?> classes;

  protected AbstractClassifier(List<?> classes) {
    this.classes = Objects.requireNonNull(classes);
  }

  @Override
  public final List<?> getClasses() {
    return classes;
  }

  @Override
  public Output<Object> predict(Input<? extends In> x) {
    Object[] labels = new Object[x.size()];
    IntStream.range(0, x.size()).parallel().forEach(i -> labels[i] = predict(x.get(i)));
    return Outputs.asOutput(labels);
  }

  @Override
  public Object predict(In input) {
    return getClasses().get(Arrays.argmax(estimate(input)));
  }

  @Override
  public DoubleArray estimate(Input<? extends In> x) {
    DoubleArray estimations = DoubleArray.zeros(x.size(), getClasses().size());
    IntStream.range(0, x.size()).parallel().forEach(i -> estimations.setRow(i, estimate(x.get(i))));
    return estimations;
  }

  @Override
  public Set<Characteristic> getCharacteristics() {
    return Collections.emptySet();
  }
}
