package org.briljantframework.mimir.data.transform;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.Well1024a;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Inputs;
import org.briljantframework.mimir.data.Instance;
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
    train = Inputs.newInput(getBuilder().set("a", Vector.fromSupplier(distribution::sample, 100))
        .set("b", Vector.fromSupplier(distribution::sample, 100))
        .set("c", Vector.fromSupplier(distribution::sample, 100)).build());

    test = Inputs.newInput(getBuilder().set("a", Vector.fromSupplier(distribution::sample, 100))
        .set("c", Vector.fromSupplier(distribution::sample, 100))
        .set("b", Vector.fromSupplier(distribution::sample, 80)).build());

  }

  private DataFrame.Builder getBuilder() {
    return DataFrame.builder();
  }
}
