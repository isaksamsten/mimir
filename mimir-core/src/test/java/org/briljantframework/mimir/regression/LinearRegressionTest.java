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

import org.briljantframework.mimir.data.*;
import org.briljantframework.mimir.supervised.data.Instance;
import org.briljantframework.mimir.supervised.data.MultidimensionalSchema;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by isak on 2017-03-01.
 */
public class LinearRegressionTest {

  @Test
  public void testLinearRegression() throws Exception {
    MultidimensionalSchema schema = new MultidimensionalSchema(3, 0);
    Input<Instance> input = schema.newInput();

    input.add(schema.newInstance().set(0, 1).set(1, 2).set(2, 3).build());
    input.add(schema.newInstance().set(0, 0).set(1, 5).set(2, 1).build());
    input.add(schema.newInstance().set(0, 1).set(1, 0.3).set(2, 5).build());
    input.add(schema.newInstance().set(0, 0).set(1, 3).set(2, 9).build());

    List<Double> output = new ArrayList<>();
    output.add(0.32);
    output.add(1.32);
    output.add(3.2);
    output.add(2.3);

    LinearRegression.Learner learner = new LinearRegression.Learner();
    LinearRegression fit = learner.fit(input, output);

    System.out.println(fit.predict(schema.newInstance().set(0, 1).set(1, 2).set(2, 3).build()));

  }
}