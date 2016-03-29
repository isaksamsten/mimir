package org.briljantframework.mimir.data.transform;

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.briljantframework.mimir.data.Input;
import org.briljantframework.mimir.data.Inputs;
import org.briljantframework.mimir.data.Instance;
import org.junit.Test;

/**
 * Created by isak on 3/29/16.
 */
public class ZNormalizerTest {

  @Test
  public void fit() throws Exception {
    DataFrame x = Datasets.loadIris();
    Input<Instance> in = Inputs.newInput(x);

    Transformation<Instance, Instance> transformation = new ZNormalizer<>();
    Transformer<Instance, Instance> transformer = transformation.fit(in);


    System.out.println(transformer.transform(in));


  }
}
