package org.mimirframework.examples;

import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.mimirframework.classification.ClassifierValidator;
import org.mimirframework.classification.EnsembleEvaluator;
import org.mimirframework.classification.HyperPlaneTree;
import org.mimirframework.classification.RandomForest;
import org.mimirframework.evaluation.Result;
import org.mimirframework.evaluation.Validator;

/**
 * @author Isak Karlsson
 */
public class HyperPlaneForestExample {
  public static void main(String[] args) {

    DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris());
    DataFrame x = iris.drop("Class").apply(v -> v.set(v.where(Is::NA), v.mean()));
    Vector y = iris.get("Class");

    Validator<RandomForest> cv = ClassifierValidator.crossValidator(10);
    cv.add(EnsembleEvaluator.INSTANCE);

    RandomForest.Configurator configurator = new RandomForest.Configurator(100);
    configurator
        .setBaseLearner(((classSet, classes) -> new HyperPlaneTree.Learner(classSet, classes, 10)));
    RandomForest.Learner hpRf = configurator.configure();

    RandomForest.Learner rf = configurator.setBaseLearner(null).configure();
    Result hpR = cv.test(hpRf, x, y);
    Result rfR = cv.test(rf, x, y);
    System.out.println(hpR.getMeasures().mean().minus(rfR.getMeasures().mean()));

  }
}
