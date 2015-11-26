package org.mimirframework.classification.conformal;

import org.briljantframework.array.DoubleArray;
import org.briljantframework.data.vector.Vector;
import org.junit.Assert;
import org.junit.Test;

/**
 * @author Isak Karlsson <isak-kar@dsv.su.se>
 */
public class ProbabilityCostFunctionTest {

  @Test
  public void testMargin() throws Exception {
    DoubleArray x = DoubleArray.of(0, 0, 1, 1, 0.9, 0, 0, 0.1, 0).reshape(3, 3);
    ProbabilityCostFunction margin = ProbabilityCostFunction.margin();
    DoubleArray m =
        ProbabilityCostFunction.estimate(margin, x, Vector.of(1, 1, 1), Vector.of(0, 1, 2));
    Assert.assertArrayEquals(DoubleArray.of(0, 0.1, 1.0).data(), m.data(), 0.01);
  }

  @Test
  public void testInverseProbability() throws Exception {

  }
}
