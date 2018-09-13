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
package org.briljantframework.mimir.data.transform;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.Well1024a;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.series.Series;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Inputs;
import org.briljantframework.mimir.supervised.data.Instance;
import org.junit.Before;

/**
 * Created by isak on 3/22/16.
 */
public class TransformationTest {
  private Input<Instance> train;
  private Input<Instance> test;

  @Before
  public void setUp() throws Exception {
    NormalDistribution distribution = new NormalDistribution(new Well1024a(100), 10, 2,
        NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
//    train = Inputs.asInput(getBuilder().setColumn("a", Series.generate(distribution::sample, 100))
//        .setColumn("b", Series.generate(distribution::sample, 100))
//        .setColumn("c", Series.generate(distribution::sample, 100)).build());
//
//    test = Inputs.asInput(getBuilder().setColumn("a", Series.generate(distribution::sample, 100))
//        .setColumn("c", Series.generate(distribution::sample, 100))
//        .setColumn("b", Series.generate(distribution::sample, 80)).build());

  }

  private DataFrame.Builder getBuilder() {
    return DataFrame.newBuilder();
  }
}
