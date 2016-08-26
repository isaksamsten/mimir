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

import java.util.List;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.series.Series;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * <p>
 * A classifier is a function {@code f(X, y)} which produces a hypothesis {@code g = X -> y} that as
 * accurately as possible model the true function {@code h} used to generate {@code X -> y}.
 * </p>
 *
 * <p>
 * The input {@code x} is usually denoted as the instances and the output {@code y} as the classes.
 * The input instances is represented as a {@link DataFrame} which consists of possibly
 * heterogeneous vectors of values characterizing each instance.
 * </p>
 *
 * <p>
 * The output of the classifier is a {@link Classifier} (i.e., the {@code g}) which (hopefully)
 * approximates {@code h}. To estimate how well {@code g} approximates {@code h}, cross-validation
 * {@link ClassifierValidator#crossValidator(int)} can be employed.
 * </p>
 *
 * A classifier is always atomic, i.e. does not have mutable state.
 *
 * <pre>
 * Tuners.crossValidation(new RandomForest.Builder(), x, y,
 *     Configuration.measureComparator(Accuracy.class), 10,
 *     range(&quot;No. trees&quot;, RandomForest.Builder::withSize, 10, 1000, 10),
 *     range(&quot;No. features&quot;, RandomForest.Builder::withMaximumFeatures, 2, x.columns(), 1));
 * </pre>
 *
 * @author Isak Karlsson
 */
public interface Classifier<In, Out> extends Predictor<In, Out> {

  /**
   * The classes this predictor is able to predict, i.e. its co-domain. Note that the i:th element
   * of the returned vector is the label of the j:th column in the probability matrix returned by
   * {@link #estimate(Input)}.
   *
   * @return the list of classes.
   */
  List<Out> getClasses();

  /**
   * Estimates the posterior probabilities for all objects in the given input
   *
   * <p>
   * Each column corresponds to the probability of a particular class (the j:th column correspond to
   * the j:th element (using {@link Series#loc()}) in {@linkplain #getClasses()}) and each row
   * corresponds to a particular record in the supplied data frame.
   *
   * @param x the data frame of records to estimate the posterior probabilities for
   * @return a matrix with probability estimates; shape = {@code [x.rows(), getClasses().size()]}.
   */
  DoubleArray estimate(Input<? extends In> x);

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
