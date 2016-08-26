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
package org.briljantframwork.mimir.image.distance;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.distance.Distance;
import org.briljantframework.mimir.distance.EuclideanDistance;
import org.junit.Test;

/**
 * Created by isak on 3/31/16.
 */
public class ImageEuclideanDistanceTest {

  @Test
  public void testCompute() throws Exception {
    DoubleArray a = DoubleArray.of(255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0)
        .reshape(4, 4);
    DoubleArray b = DoubleArray.of(0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255)
        .reshape(4, 4);
    Distance<DoubleArray> distance = new ImageEuclideanDistance();
    System.out.println(distance.compute(a, b));
    System.out.println(EuclideanDistance.getInstance().compute(a, b));
  }
}
