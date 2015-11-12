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

package org.briljantframework.regression;

import org.briljantframework.Check;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.supervised.Predictor;

/**
 * @author Isak Karlsson
 */
public interface Regression extends Predictor {

  /**
   * Predict the value for the given input vector
   * 
   * @param y the input vector
   * @return the predicted value
   */
  double predict(Vector y);

  /**
   * @author Isak Karlsson <isak-kar@dsv.su.se>
   */
  interface Learner<P extends Regression> extends Predictor.Learner<P> {

    default P fit(DataFrame x, Vector y) {
      Check.argument(x.rows() == y.size(), "Size of input data and input target don't match");
      Check.argument(x.getColumns().stream().allMatch(Is::numeric),
          "Only supports numerical data.");
      Check.argument(Is.numeric(y), "Only support numerical target");
      return fit(x.toDoubleArray(), y.toDoubleArray());
    }

    P fit(DoubleArray x, DoubleArray y);
  }
}
