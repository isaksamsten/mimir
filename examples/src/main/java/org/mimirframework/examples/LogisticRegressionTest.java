package org.mimirframework.examples;

import org.briljantframework.data.Is;
import org.briljantframework.data.dataframe.DataFrame;
import org.briljantframework.data.dataframe.DataFrames;
import org.briljantframework.data.vector.Vector;
import org.briljantframework.dataset.io.Datasets;
import org.mimirframework.classification.Classifier;
import org.mimirframework.classification.ClassifierValidator;
import org.mimirframework.classification.LogisticRegression;
import org.mimirframework.evaluation.Result;

/**
 * @author Isak Karlsson
 */
public class LogisticRegressionTest {
  public static void main(String[] args) {
    DataFrame iris = DataFrames.permuteRecords(Datasets.loadIris());
    DataFrame x = iris.drop("Class").apply(v -> v.set(v.where(Is::NA), v.mean()));
    Vector y = iris.get("Class");

    // Multinomial logistic regression (handles |unique(y)| > 1)
    Classifier.Learner<LogisticRegression> classifier = new LogisticRegression.Learner();
    Result result = ClassifierValidator.crossValidation(10).test(classifier, x, y);
    DataFrame measures = result.getMeasures();
    System.out.println(measures.mean());

    testOdds();
  }

  public static void testOdds()  {
    // Construct a dataset
    DataFrame x = DataFrame.of("Age", Vector.of(55, 28, 65, 46, 86, 56, 85, 33, 21, 42), "Smoker",
        Vector.of(0, 0, 1, 0, 1, 1, 0, 0, 1, 1));

    // Construct a target (got cancer / not)
    Vector y = Vector.of(0, 0, 0, 1, 1, 1, 0, 0, 0, 1);
    System.out.println(x.set("Cancer?", y));

    LogisticRegression.Learner regression = new LogisticRegression.Learner();
    LogisticRegression model = regression.fit(x, y);
    System.out.println(model);

    System.out.println("(Intercept) " + model.getOddsRatio("(Intercept)"));
    for (Object o : x.getColumnIndex().keySet()) {
      System.out.println(o + " " + model.getOddsRatio(o));
    }
  }
}
