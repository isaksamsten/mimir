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

import org.briljantframework.array.Array;
import org.briljantframework.data.dataframe.DataFrame;
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
 * approximates {@code h}. To estimate how well {@code g} approximates {@code h},
 * {@link ClassifierValidator#crossValidator(int) cross-validation} can be employed.
 * </p>
 *
 * A classifier is always atomic, i.e. does not have mutable state.
 *
 * @author Isak Karlsson
 */
public interface Classifier<In, Out> extends Predictor<In, Out> {

  /**
   * The classes this classifier is able to predict/produce from its {@link #predict(Object) predict
   * function}, i.e. the classifiers co-domain.
   *
   * @return the array of class labels.
   */
  Array<Out> getClasses();
}
