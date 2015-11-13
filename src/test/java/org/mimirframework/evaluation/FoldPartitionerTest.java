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

package org.mimirframework.evaluation;

import static org.junit.Assert.assertTrue;

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.junit.Test;
import org.mimirframework.evaluation.partition.FoldPartitioner;
import org.mimirframework.evaluation.partition.Partition;
import org.mimirframework.evaluation.partition.Partitioner;

public class FoldPartitionerTest {

  @Test
  public void testPartition() throws Exception {
    DataFrame dummy = Datasets.loadIris();
    DataFrame x = dummy.loc().get(0, 1, 2, 3);
    Vector y = dummy.loc().get(4);

    System.out.println(x);

    Partitioner strategy = new FoldPartitioner(10);
    Iterable<Partition> partitionIterator = strategy.partition(x, y);
    int i = 0;
    for (Partition partition : partitionIterator) {
      System.out.println("Fold " + i++);
      System.out.println(partition.getTrainingData().limit(135));
      System.out.println(partition.getValidationData().limit(15));
    }

    // DataFrame iris = DataFrames.permuteRows(Datasets.loadIris());
    // DataFrame irisX = iris.takeColumns(IntRange.range(0, 4));
    // Vector irisY = iris.getColumn(4);
    //
    // RandomForest tree = RandomForest.withSize(100).build();
    //
    // long start = System.currentTimeMillis();
    // System.out.println(ClassificationEvaluators.crossValidation(10).acceptEvaluators(tree, irisX,
    // irisY));
    // System.out.println(System.currentTimeMillis() - start);

    // DataFrame synthetic = Datasets.loadSyntheticControl();
    // DataFrame x = synthetic.takeColumns(IntRange.range(1, synthetic.columns()));
    // Vector y = synthetic.getColumnView(0);
    //
    // RandomShapeletForest forest =
    // RandomShapeletForest.withSize(10).withInspectedShapelets(100).create();
    // System.out.println(Evaluators.crossValidation(10).acceptEvaluators(forest, x, y));
    //
    assertTrue(true);
  }
}
