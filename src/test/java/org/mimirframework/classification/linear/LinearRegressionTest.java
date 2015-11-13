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

package org.mimirframework.classification.linear;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.mimirframework.regression.LinearRegression;
import org.junit.Test;

public class LinearRegressionTest {

  @Test
  public void testFit() throws Exception {
    DoubleArray x = Arrays.randn(100).reshape(50, 2);
    DoubleArray y = sinTarget(x);

    LinearRegression.Learner learner = new LinearRegression.Learner();
    LinearRegression regression = learner.fit(x, y);
    System.out.println(regression.getTheta());
  }

  private DoubleArray sinTarget(DoubleArray x) {
    return x.getColumn(0).minus(x.getColumn(1)).plus(x.getColumn(0).map(i -> Math.sin(i) * Math.PI))
        .map(Math::signum);
  }

}
