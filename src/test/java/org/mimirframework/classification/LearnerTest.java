package org.mimirframework.classification;

import java.io.FileInputStream;

import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.dataframe.MixedDataFrame;
import org.briljantframework.data.dataframe.transform.MinMaxNormalizer;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.briljantframework.dataset.io.RdsDatasetReader;
import org.junit.Test;
import org.mimirframework.evaluation.Result;
import org.mimirframework.evaluation.Validator;

/**
 * Created by isak on 11/16/15.
 */
public class LearnerTest {

  @Test
  public void testFit() throws Exception {
    // DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris());
    // DataFrame x = iris.drop("Class").apply(v -> v.set(v.where(Is::NA), v.mean()));
    // Vector y = iris.get("Class");
    // DataFrame data = DataFrames.permuteRecords(Datasets.loadSyntheticControl());
    // DataFrame x = data.drop(0);
    // Vector y = data.get(0);

    DataFrame data = DataFrames.permuteRecords(Datasets.load(MixedDataFrame.Builder::new,
        new RdsDatasetReader(new FileInputStream("/home/isak/Desktop/glass.txt"))));

    String classColumn = "Type";
    DataFrame x = data.drop(classColumn).apply(v -> v.set(v.where(Is::NA), v.mean()));
    x = new MinMaxNormalizer().fitTransform(x);
    Vector y = data.get(classColumn);
    Validator<RandomForest> cv = ClassifierValidator.crossValidation(10);
    cv.add(EnsembleEvaluator.INSTANCE);

    RandomForest.Configurator configurator = new RandomForest.Configurator(100);
    configurator.setBaseLearner((set, classes) -> new HyperPlaneTree.Learner(set, classes, 100));
    RandomForest.Learner rf = configurator.configure();

    Result result = cv.test(rf, x, y);
    System.out.println(result.getMeasures().mean());

  }
}
