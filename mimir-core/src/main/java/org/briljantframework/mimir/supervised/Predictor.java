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

import org.briljantframework.mimir.Properties;
import org.briljantframework.mimir.data.Input;

import java.util.List;

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
  List<Out> predict(Input<In> x);

  /**
   * Generates a predictor based on some underlying algorithm and properties.
   */
  abstract class Learner<In, Out, P extends Predictor<In, Out>> extends Parameterized {

    protected Learner(Properties parameters) {
      super(parameters);
    }

    protected Learner() {}

    /**
     * Fit a predictor to the given input data.
     *
     * @param in the input data
     * @param out the output target
     * @return a predictor for predicting the output value of an input
     */
    public abstract P fit(Input<In> in, List<Out> out);

    @Override
    public String toString() {
      return "Learner{ parameters: " + getParameters() + "}";
    }
  }
}
