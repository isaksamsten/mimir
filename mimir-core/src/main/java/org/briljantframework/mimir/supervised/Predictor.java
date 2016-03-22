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

import java.util.Collection;
import java.util.Collections;
import java.util.Set;

import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Output;
import org.briljantframework.mimir.data.Property;

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
   * A Learner represents a generator predictors.
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
     * Reports the requirements imposed on the input.
     * 
     * @return a collection of input properties.
     */
    default Collection<Property<?>> getRequiredInputProperties() {
      return Collections.emptySet();
    }
  }

  interface Configurator<In, Out, C extends Learner<In, Out, ? extends Predictor<In, Out>>> {
    C configure();
  }
}
