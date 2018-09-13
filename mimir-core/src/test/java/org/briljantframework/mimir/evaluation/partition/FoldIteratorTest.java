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
package org.briljantframework.mimir.evaluation.partition;

import java.util.List;

import org.briljantframework.array.Arrays;
import org.briljantframework.array.DoubleArray;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.supervised.data.Instance;
import org.briljantframework.mimir.supervised.data.MultidimensionalSchema;
import org.junit.Test;

/**
 * Created by isak on 2017-03-14.
 */
public class FoldIteratorTest {


  @Test
  public void testFindSubGraph() throws Exception {
    //@formatter:off
    DoubleArray graph = Arrays.doubleMatrix(new double[][]{
        // A  B  C  D
          {0, 1, 0, 1},
          {1, 0, 0, 0},
          {0, 0, 0, 1},
          {1, 0, 1, 0}
    });
    //@formatter:on

    //@formatter:off
    DoubleArray sub = Arrays.doubleMatrix(new double[][]{
        // A  B  C  D
          {0, 1, 0, 1},
          {1, 0, 0, 0},
          {0, 0, 0, 0},
          {1, 0, 0, 0}
    });
    //@formatter:on


    contains(graph, sub);


  }

  private void contains(DoubleArray graph, DoubleArray sub) {
    System.out.println(Arrays.sum(Arrays.times(graph, sub)) / Arrays.sum(sub));
  }

  private int findFirst(DoubleArray sub) {
    for (int i = 0; i < sub.size(); i++) {
        if (sub.get(i) > 0) {
          return i;
        }
    }

    return -1;
  }

  @Test
  public void testRandomOrder() throws Exception {
    MultidimensionalSchema schema = new MultidimensionalSchema(1, 0);
    Input<Instance> input = schema.newInput();
    input.add(schema.newInstance().set(0, 1).build());
    input.add(schema.newInstance().set(0, 2).build());
    input.add(schema.newInstance().set(0, 3).build());
    input.add(schema.newInstance().set(0, 4).build());
    input.add(schema.newInstance().set(0, 5).build());
    input.add(schema.newInstance().set(0, 6).build());
    input.add(schema.newInstance().set(0, 7).build());

    List<Object> output = java.util.Arrays.asList(0, 0, 0, 0, 1, 1, 1);


    FoldIterator<Instance, Object> iterator = new FoldIterator<>(input, output, 3, true);
    while (iterator.hasNext()) {
      Partition<Instance, Object> next = iterator.next();
      System.out.println(next.getTrainingData() +" :: " + next.getValidationData());
    }
  }
}
