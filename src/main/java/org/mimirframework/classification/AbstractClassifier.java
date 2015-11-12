/*
 * The MIT License (MIT)
 * 
 * Copyright (c) 2015 Isak Karlsson
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

package org.mimirframework.classification;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Objects;
import java.util.Set;
import java.util.stream.IntStream;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.index.VectorLocationSetter;
import org.briljantframework.data.vector.Vector;
import org.mimirframework.supervised.Characteristic;

/**
 * Provides sane defaults for a predictor. Sub-classes only have to implement the
 * {@link #estimate(org.briljantframework.data.vector.Vector)} method to have a sensible default
 * predictor.
 * 
 * <p/>
 * For a classifier unable to output probability estimates the {@linkplain #estimate(Vector)} should
 * return an array where the predicted class has probability {@code 1} and all other classes
 * probability {@code 0}. To improve performance, implementors of such predictors should consider
 * overriding the default {@linkplain #predict(Vector)} method.
 * 
 * <p/>
 * Predictors that produces probability estimates should make sure to include the
 * {@link ClassifierCharacteristic#ESTIMATOR ESTIMATOR} characteristics in the {@link EnumSet}
 * returned by {@link #getCharacteristics()}
 *
 * @author Isak Karlsson
 */
public abstract class AbstractClassifier implements Classifier {

  private final Vector classes;

  protected AbstractClassifier(Vector classes) {
    this.classes = Objects.requireNonNull(classes);
  }

  @Override
  public final Vector getClasses() {
    return classes;
  }

  @Override
  public Vector predict(DataFrame x) {
    // This is really only safe since Builder is initialized with a size i.e. filled with NA
    Vector.Builder labels = Vector.Builder.withSize(Object.class, x.rows());
    VectorLocationSetter loc = labels.loc();
    IntStream.range(0, x.rows()).parallel().forEach(i -> loc.set(i, predict(x.loc().getRecord(i))));
    return labels.build();
  }

  @Override
  public Object predict(Vector record) {
    return getClasses().loc().get(Object.class, Arrays.argmax(estimate(record)));
  }

  @Override
  public DoubleArray estimate(DataFrame x) {
    DoubleArray estimations = Arrays.newDoubleArray(x.rows(), getClasses().size());
    IntStream.range(0, x.rows()).parallel()
        .forEach(i -> estimations.setRow(i, estimate(x.loc().getRecord(i))));
    return estimations;
  }

  @Override
  public Set<Characteristic> getCharacteristics() {
    return Collections.emptySet();
  }
}
