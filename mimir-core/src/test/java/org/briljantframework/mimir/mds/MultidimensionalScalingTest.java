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
package org.briljantframework.mimir.mds;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.data.ArrayInput;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.transform.Transformers;
import org.briljantframework.mimir.distance.EuclideanDistance;
import org.junit.Test;

/**
 * Created by isak on 2016-09-28.
 */
public class MultidimensionalScalingTest {


  @Test
  public void testTransformy() throws Exception {
    Input<DoubleArray> thing = new ArrayInput<>();
    thing.add(DoubleArray.of(1, 2, 3));
    thing.add(DoubleArray.of(1, 2, 3));
    thing.add(DoubleArray.of(1, 2, 3));


    MultidimensionalScaling t2 = new MultidimensionalScaling(2);


  }
}