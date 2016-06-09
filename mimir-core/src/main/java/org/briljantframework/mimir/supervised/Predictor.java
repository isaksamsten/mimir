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
package org.briljantframework.mimir.supervised;

import java.util.Set;

import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Output;
import org.briljantframework.mimir.data.TypeKey;
import org.briljantframework.mimir.data.TypeMap;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public interface Predictor<In, Out> {

  /**
   * Predict the output given the input.
   *
   * @param in the input
   * @return the predicted output
   */
  Out predict(In in);

  /**
   * Return a vector of predictions for the rows in the given data frame.
   * 
   * @param x the data frame
   * @return a vector of predictions
   */
  Output<Out> predict(Input<? extends In> x);

  /**
   * Get a set of characteristics for this particular predictor
   *
   * @return the set of characteristics
   */
  Set<Characteristic> getCharacteristics();

  /**
   * Generates a predictor based on some underlying algorithm and properties.
   */
  interface Learner<In, Out, P extends Predictor<In, Out>> {

    /**
     * Fit a predictor to the given input data.
     * 
     * @param in the input data
     * @param out the output target
     * @return a predictor for predicting the output value of an input
     */
    P fit(Input<? extends In> in, Output<? extends Out> out);

    /**
     * Set the specified parameter to the specified value.
     *
     * @param key the parameter key
     * @param value the parameter value
     * @param <T> the type of parameter value
     */
    <T> void set(TypeKey<T> key, T value);

    /**
     * Get the parameter value for the specified key.
     *
     * @param key the parameter key
     * @param <T> the type of parameter value
     * @return the parameter value
     */
    <T> T get(TypeKey<T> key);

    /**
     * Get a typed map of parameter values.
     *
     * @return the parameters
     */
    TypeMap getParameters();
  }

  @Deprecated
  interface Configurator<In, Out, C extends Learner<In, Out, ? extends Predictor<In, Out>>> {
    C configure();
  }
}
