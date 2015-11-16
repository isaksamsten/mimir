package org.mimirframework.classification;

import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.junit.Test;
import org.mimirframework.evaluation.Result;
import org.mimirframework.evaluation.Validator;

/**
 * Created by isak on 11/16/15.
 */
public class LearnerTest {

  @Test
  public void testFit() throws Exception {
    DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris());
    DataFrame x = iris.drop("Class").apply(v -> v.set(v.where(Is::NA), v.mean()));
    Vector y = iris.get("Class");

    Validator<RandomForest> cv = ClassifierValidator.crossValidation(10);
    cv.add(EnsembleEvaluator.INSTANCE);

    RandomForest.Configurator configurator = new RandomForest.Configurator(100);
    configurator.setBaseLearner(HyperPlaneTree.Learner::new);
    RandomForest.Learner rf = configurator.configure();

    Result result = cv.test(rf, x, y);
    System.out.println(result.getMeasures().mean());

  }
}
