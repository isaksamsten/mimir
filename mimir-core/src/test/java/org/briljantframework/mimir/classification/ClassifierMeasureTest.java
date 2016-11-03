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
package org.briljantframework.mimir.classification;

import static org.junit.Assert.assertEquals;

import org.briljantframework.array.Array;
import org.briljantframework.data.series.Series;
import org.briljantframework.mimir.data.ArrayOutput;
import org.junit.Test;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ClassifierMeasureTest {

  @Test
  public void testGetPrecision() throws Exception {
    Series t = Series.of(1, 1, 1, 2);
    Series p = Series.of(1, 1, 2, 1);
    ClassifierMeasure cm = new ClassifierMeasure(Array.of(1, 2), new ArrayOutput<>(p.values()),
        new ArrayOutput<>(t.values()));
    assertEquals(0.333, cm.getPrecision(), 0.01);
    assertEquals(0.333, cm.getRecall(), 0.01);
    assertEquals(0.5, cm.getAccuracy(), 0.01);
  }
}
