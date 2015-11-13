package org.mimirframework.examples;

import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.mimirframework.classification.Classifier;
import org.mimirframework.classification.ClassifierValidator;
import org.mimirframework.classification.LogisticRegression;
import org.mimirframework.classification.RandomForest;
import org.mimirframework.evaluation.Result;

/**
 * @author Isak Karlsson
 */
public class LogisticRegressionTest {
  public static void main(String[] args) {
    DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris());
    DataFrame x = iris.drop("Class");
    Vector y = iris.get("Class");
    Classifier.Learner<? extends Classifier> classifier = new RandomForest.Learner(100);
    ClassifierValidator<Classifier> cv = ClassifierValidator.crossValidation(10);
    Result result = cv.test((d, t) -> classifier.fit(x, t), x, y);
    System.out.println(result.getMeasures().mean());
  }

  public static void testOdds() throws Exception {
    DataFrame x = DataFrame.of("Age", Vector.of(55, 28, 65, 46, 86, 56, 85, 33, 21, 42), "Smoker",
        Vector.of(0, 0, 1, 0, 1, 1, 0, 0, 1, 1));
    Vector y = Vector.of(0, 0, 0, 1, 1, 1, 0, 0, 0, 1);
    System.out.println(x);

    LogisticRegression.Learner regression = new LogisticRegression.Learner();
    LogisticRegression model = regression.fit(x, y);
    System.out.println(model);

    System.out.println("(Intercept) " + model.getOddsRatio("(Intercept)"));
    for (Object o : x.getColumnIndex().keySet()) {
      System.out.println(o + " " + model.getOddsRatio(o));
    }
  }
}
