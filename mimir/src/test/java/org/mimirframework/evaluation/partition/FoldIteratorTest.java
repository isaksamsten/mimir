package org.mimirframework.evaluation.partition;

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.IntVector;
import org.briljantframework.data.vector.Vector;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class FoldIteratorTest {

  @Test
  public void testFoldIterator() throws Exception {
    DataFrame x = DataFrame.of("A", IntVector.range(20), "B", IntVector.range(20));
    Vector y = IntVector.range(20);
    FoldPartitioner partitioner = new FoldPartitioner(10);
    for (Partition partition : partitioner.partition(x, y)) {
      System.out.println(partition.getTrainingData());
      System.out.println(partition.getValidationData());
    }



  }
}
