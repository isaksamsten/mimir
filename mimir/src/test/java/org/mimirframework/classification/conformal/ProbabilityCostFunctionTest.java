package org.mimirframework.classification.conformal;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.vector.Vector;
import org.junit.Test;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ProbabilityCostFunctionTest {

  @Test
  public void testMargin() throws Exception {
    DoubleArray x = DoubleArray.of(0, 0, 1, 1, 0.9, 0, 0, 0.1, 0).reshape(3, 3);
    DoubleArray m =
        ProbabilityCostFunction.margin().apply(x, Vector.of(1, 1, 1), Vector.of(0, 1, 2));
    System.out.println(m);
  }

  @Test
  public void testInverseProbability() throws Exception {

  }
}
