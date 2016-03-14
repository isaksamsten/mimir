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
package org.briljantframework.mimir.regression;

import java.util.Collections;
import java.util.Set;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.mimir.*;
import org.briljantframework.mimir.supervised.Characteristic;
import org.briljantframework.mimir.supervised.Predictor;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public final class LinearRegression implements Regression<Instance> {

  private final DoubleArray theta;

  private LinearRegression(DoubleArray theta) {
    this.theta = theta;
  }

  public DoubleArray getParameters() {
    return theta.copy();
  }

  @Override
  public Double predict(Instance y) {
    return 0.0;
  }

  @Override
  public Output<Double> predict(Input<? extends Instance> x) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Set<Characteristic> getCharacteristics() {
    return Collections.emptySet();
  }

  /**
   * @author Isak Karlsson
   */
  public static class Learner implements Predictor.Learner<Instance, Double, LinearRegression> {

    public Learner() {}

    @Override
    public LinearRegression fit(Input<? extends Instance> in, Output<? extends Double> out) {
      DoubleArray x = Inputs.toDoubleArray(in);
      DoubleArray y = Outputs.toDoubleArray(out);


      x = Arrays.vstack(Arrays.ones(x.rows()), x);
      DoubleArray inner =
          Arrays.dot(Arrays.dot(x.transpose(), x), Arrays.eye(x.columns()).times(1));
      DoubleArray v = Arrays.dot(Arrays.linalg.pinv(inner), x.transpose());
      DoubleArray theta = Arrays.dot(v, y);

      // X'(XX'\lambda*I_n)^-1y
      return new LinearRegression(theta);
    }

  }
}
