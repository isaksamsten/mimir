package org.mimirframework.examples;

import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.mimirframework.classification.ClassifierValidator;
import org.mimirframework.classification.EnsembleEvaluator;
import org.mimirframework.classification.RandomForest;
import org.mimirframework.evaluation.Result;
import org.mimirframework.evaluation.Validator;

/**
 * @author Isak Karlsson
 */
public class RandomForestExample {
  public static void main(String[] args) {
    DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris());
    DataFrame x = iris.drop("Class").apply(v -> v.set(v.where(Is::NA), v.mean()));
    Vector y = iris.get("Class");

    Validator<RandomForest> cv = ClassifierValidator.crossValidator(10);
    cv.add(EnsembleEvaluator.INSTANCE);

    Result result = cv.test(new RandomForest.Learner(100), x, y);
    System.out.println(result.getMeasures().mean());


    /*
     * for (int i = 0; i < f.size(); i++) { RandomForest.Learner forest = new
     * RandomForest.Configurator(100).setMaximumFeatures(f.get(i)).configure(); Result result =
     * classifierValidator.test(forest, x, y); System.out.println(result.getMeasures()); }
     */
  }
}
